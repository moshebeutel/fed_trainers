import copy
import logging
from collections import OrderedDict
import numpy as np
import torch
from tqdm import trange
from fed_trainers.trainers.model import get_model
from fed_trainers.trainers.utils import get_clients, get_device, local_train, flatten_tensor, eval_model, update_frame, log2wandb, \
    load_aggregated_grads_to_global_net, wandb_plot_confusion_matrix


def train(args, dataloaders):
    logger = logging.getLogger(args.log_name)

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
        # Initialize global model params
        grads = OrderedDict()
        prev_params = OrderedDict()
        for n, p in net.named_parameters():
            grads[n] = []
            prev_params[n] = p.detach()

        # Sample several clients
        client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)

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

            # get client grads
            for n, p in local_net.named_parameters():
                grads[n].append(p.data.detach() - prev_params[n])

        # stack sampled clients grads
        grads_list = [torch.stack(grads[n]) for n, p in net.named_parameters()]

        # flatten grads for clipping and noising
        grads_flattened = flatten_tensor(grads_list)

        # clip grads
        grads_norms = torch.norm(grads_flattened, p=2, dim=-1)
        clip_factor = torch.max(torch.ones_like(grads_norms), grads_norms / args.clip)
        grads_flattened_clipped = torch.div(grads_flattened, clip_factor.reshape(-1, 1))

        # noise grads
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                             size=grads_flattened_clipped.shape).to(device)
        noised_grads = grads_flattened_clipped + noise

        # aggregate noised grads
        aggregated_grads = noised_grads.mean(dim=0)

        # update global net
        global_lr = args.global_lr ** step
        logger.debug(f'Global learning rate: {global_lr}')
        net = load_aggregated_grads_to_global_net(aggregated_grads, net, prev_params, global_lr)

        # Evaluate model
        if ((step + 1) > args.eval_after and (step + 1) % args.eval_every == 0) or (step + 1) == args.num_steps:

            val_results = eval_model(args, net, private_clients, val_loaders)

            val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict, \
                val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results

            if val_avg_acc > best_acc:
                best_acc = val_avg_acc
                best_loss = val_avg_loss
                best_acc_score = val_avg_acc_score
                best_f1 = val_avg_f1
                best_epoch = step
                best_model.cpu()
                del best_model
                best_model = copy.deepcopy(net)

        # Monitor using Weights & Biases
        if args.wandb:
            log2wandb(best_acc, best_acc_score, best_epoch, best_f1, best_loss, step, train_avg_loss, val_acc_dict,
                      val_acc_score_dict, val_avg_acc, val_avg_acc_score, val_avg_f1, val_avg_loss, val_f1s_dict,
                      val_loss_dict)

    # calibration
    for j, c_id in enumerate(private_clients):
        calib_loader = val_loaders[c_id]

        pbar_dict.update(
            {
                'Step': 'Cal',
                'Client': f'{c_id}'.zfill(3),
                'Client Number in Step': f'{(j + 1)}'.zfill(3),
                # 'Train Avg Loss': f'{train_avg_loss:.4f}',
                # 'Train Current Loss': f'{0.:.4f}',
                # 'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                # 'Val Avg Acc': f'{val_avg_acc:.4f}',
                # 'Best Avg Acc': f'{best_acc:.4f}'})
            })
        local_net, clib_avg_loss = local_train(args, net, calib_loader,
                                               pbar=step_iter, pbar_dict=pbar_dict)

    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders, plot_confusion_matrix=True)

    y_true_all, y_pred_all, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results
    # _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    if args.wandb:
        wandb_plot_confusion_matrix(y_true_all, y_pred_all, list(range(args.num_classes)))


    update_frame(args, dp_method='SGD_DP', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
                 test_avg_acc=test_avg_acc)
