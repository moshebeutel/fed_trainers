import copy
import logging
from collections import OrderedDict
import numpy as np
import torch
import wandb
from tqdm import trange
from model import get_model
from utils import get_clients, get_device, local_train, flatten_tensor, eval_model


def train(args, dataloaders):
    logger = logging.getLogger(args.log_name)

    fields_list = ["num_blocks", "block_size", "optimizer", "lr", "num_client_agg", "clip", "noise_multiplier"]
    args_list = [(k, vars(args)[k]) for k in fields_list]

    logger.info(f' *** Training for args {args_list} ***')

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1 = 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    public_clients, private_clients, dummy_clients = get_clients(args)
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)

    train_loaders, val_loaders, test_loaders = dataloaders

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    step_iter = trange(args.num_steps)
    pbar_dict = {'Step': '0', 'Client': '0',
                 'Client Number in Step': '0', 'Best Epoch': '0', 'Val Avg Acc': '0.0',
                 'Best Avg Acc': '0.0', 'Train Avg Loss': '0.0'}

    for step in step_iter:

        # select several clients
        client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)

        # initialize global model params
        params = OrderedDict()
        grads = OrderedDict()
        prev_params = OrderedDict()
        for n, p in net.named_parameters():
            params[n] = torch.zeros_like(p.data)
            grads[n] = []
            prev_params[n] = p.detach()

        # iterate over each client
        train_avg_loss = 0

        for j, c_id in enumerate(client_ids_step):

            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3),
                              'Client': f'{c_id}'.zfill(3),
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}',
                              'Train Current Loss': f'{0.:.4f}',
                              'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                              'Val Avg Acc': f'{val_avg_acc:.4f}',
                              'Best Avg Acc': f'{best_acc:.4f}'})

            local_net, train_avg_loss = local_train(args, net, train_loader,
                                                    pbar=step_iter, pbar_dict=pbar_dict)

            # get client grads and sum.
            for n, p in local_net.named_parameters():
                grads[n].append(p.data.detach() - prev_params[n])

        # stack sampled clients grads
        grads_list = [torch.stack(grads[n]) for n, p in net.named_parameters()]

        # flatten grads for clipping and noising
        grads_flattened = flatten_tensor(grads_list)

        # clip grads
        grads_norms = torch.norm(grads_flattened, p=2, dim=-1, keepdim=True)
        clip_factor = torch.max(torch.ones_like(grads_norms), grads_norms / args.clip)
        grads_flattened_clipped = grads_flattened / clip_factor

        # noise grads
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                             size=grads_flattened_clipped.shape).to(device)
        noised_grads = grads_flattened_clipped + noise

        # aggregate noised grads
        aggregated_grads = noised_grads.mean(dim=0)

        # update old parameters using private aggregated grads
        offset = 0
        for n, p in prev_params.items():
            params[n] = p + aggregated_grads[offset: offset + p.numel()].reshape(p.shape)
            offset += p.numel()

        # update new parameters of global net
        net.load_state_dict(params)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
            val_results = eval_model(args, net, private_clients, val_loaders)

            val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict, \
                val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results

            if val_avg_acc > best_acc:
                best_acc = val_avg_acc
                best_loss = val_avg_loss
                best_acc_score = val_avg_acc_score
                best_f1 = val_avg_f1
                best_epoch = step
                best_model = best_model.cpu()
                del best_model
                best_model = copy.deepcopy(net)

        if args.wandb:
            log_dict = {}
            log_dict.update(
                {
                    'custom_step': step,
                    'train_loss': train_avg_loss,
                    'test_avg_loss': val_avg_loss,
                    'test_avg_acc': val_avg_acc,
                    'test_avg_acc_score': val_avg_acc_score,
                    'test_avg_f1': val_avg_f1,
                    'test_best_loss': best_loss,
                    'test_best_acc': best_acc,
                    'test_best_acc_score': best_acc_score,
                    'test_best_f1': best_f1,
                    'test_best_epoch': best_epoch
                }
            )
            log_dict.update({f"test_acc_{l}": m for (l, m) in val_acc_dict.items()})
            log_dict.update({f"test_loss_{l}": m for (l, m) in val_loss_dict.items()})
            log_dict.update({f"test_acc_score_{l}": m for (l, m) in val_acc_score_dict.items()})
            log_dict.update({f"test_f1_{l}": m for (l, m) in val_f1s_dict.items()})

            wandb.log(log_dict)

        logger.debug(
            f"epoch {step}, "
            f"train loss {train_avg_loss:.3f}, "
            f"best epoch: {best_epoch}, "
            f"best test loss: {best_loss:.3f}, "
            f"best test acc: {best_acc:.3f}, "
            f"test test acc score: {best_acc_score:.3f}, "
            f"best test f1: {best_f1:.3f}"
        )

    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'### Test Results For Args {args_list}:'
                 f' test acc {test_avg_acc:.4f},'
                 f' test loss {test_avg_loss:.4f} ###')
