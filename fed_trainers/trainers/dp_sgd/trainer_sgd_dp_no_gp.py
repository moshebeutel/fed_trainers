import copy
import logging
from collections import OrderedDict
import numpy as np
import torch
from tqdm import trange

from fed_trainers.trainers.utils import (get_device, local_train, flatten_tensor, eval_model,
    # update_frame, \
                                         log2wandb, \
                                         load_aggregated_grads_to_global_net, compute_steps, compute_steps_in_epoch,
                                         logtest2wandb, wandb_plot_confusion_matrix)
from fed_trainers.trainers.factory import get_clients, get_model, get_logger


def train(args, dataloaders):
    logger = get_logger(args)

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1, train_acc_of_best_model = 0.0, 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    public_clients, private_clients, dummy_clients = get_clients(args)
    all_clients = public_clients + private_clients
    num_public_clients = len(public_clients)
    device = get_device()
    # device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)

    train_loaders, val_loaders, test_loaders = dataloaders

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    num_steps = compute_steps(args)
    logger.info(f'Num steps: {num_steps}')
    steps_in_epoch = compute_steps_in_epoch(args)
    logger.info(f'Num steps in epoch: {steps_in_epoch}')
    current_epoch_grads_norms_list = []
    current_epoch_train_avg_acc_list = []
    current_epoch_train_avg_loss_list = []
    current_epoch_val_avg_acc_list = []
    current_epoch_grads_avg_norms = (0.0, 0.0)
    current_epoch_train_avg_acc = 0.0
    current_epoch_train_avg_loss = 0.0
    current_epoch_val_avg_acc = 0.0
    step_iter = trange(num_steps)
    pbar_dict = {'Step': '0',
                 'Epoch': '0',
                 # 'Client': '0',
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
        # client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)
        client_ids_step = np.random.choice(all_clients, size=args.num_client_agg, replace=False)

        train_avg_loss, train_avg_acc = 0.0, 0.0

        logger.debug(f"Clients sampled: {client_ids_step}")
        private_clients_mask = torch.ones(size=(len(client_ids_step),), device=device)
        # Iterate over each client
        for j, c_id in enumerate(client_ids_step):
            private_clients_mask[j] = 1 if c_id in private_clients else 0
            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3),
                              # 'Client': f'{c_id}'.zfill(3),
                              'Epoch': f'{(step // steps_in_epoch) + 1}'.zfill(3),
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}',
                              'Train Current Loss': f'{0.:.2f}'.zfill(5),
                              'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                              'Val Avg Acc': f'{val_avg_acc:.4f}',
                              'Best Avg Acc': f'{best_acc:.4f}'})

            local_net, train_loss, train_acc = local_train(args, net, train_loader,
                                                           pbar=step_iter, pbar_dict=pbar_dict)

            train_avg_acc += (train_acc / args.num_client_agg)
            train_avg_loss += (train_loss / args.num_client_agg)

            # get client grads
            for n, p in local_net.named_parameters():
                grads[n].append(p.data.detach() - prev_params[n])

        current_epoch_train_avg_acc_list.append(train_avg_acc)
        current_epoch_train_avg_loss_list.append(train_avg_loss)

        # stack sampled clients grads
        grads_list = [torch.stack(grads[n]) for n, p in net.named_parameters()]

        # flatten grads for clipping and noising
        grads_flattened = flatten_tensor(grads_list)

        # clip grads
        # grads_max_amp, _ = torch.max(torch.abs(grads_flattened), dim=-1)
        grads_norms = torch.norm(grads_flattened, p=2, dim=-1)
        # args.clip = grads_norms.min() * 0.9
        clip_factor = torch.max(torch.ones_like(grads_norms), grads_norms / args.clip)
        grads_flattened_clipped = torch.div(grads_flattened, clip_factor.reshape(-1, 1))

        current_epoch_grads_norms_list.append((float(grads_norms.mean()),
                                               float(torch.norm(grads_flattened_clipped, p=2, dim=-1).mean())))

        # noise grads
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                             size=grads_flattened_clipped.shape).to(device)
        # Add noise to grads. Note: public clients add zero noise
        noised_grads = grads_flattened_clipped + noise * private_clients_mask.unsqueeze(1)

        # aggregate noised grads
        aggregated_grads = noised_grads.mean(dim=0)

        # update global net
        global_lr = args.global_lr

        net = load_aggregated_grads_to_global_net(aggregated_grads, net, prev_params, global_lr)

        # Evaluate model
        if ((step + 1) > args.eval_after and (step + 1) % args.eval_every == 0) or (step + 1) == num_steps:
            val_results = eval_model(args, net, private_clients, val_loaders, plot_confusion_matrix=True)
            y_true_all, y_pred_all, _, _, val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results
            if args.wandb:
                wandb_plot_confusion_matrix(y_true_all, y_pred_all, list(range(args.num_classes)))


            # val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict, \
            # val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results

            current_epoch_val_avg_acc_list.append(val_avg_acc)
            if len(current_epoch_val_avg_acc_list) >= (float(steps_in_epoch) / float(args.eval_every)):
                current_epoch_train_avg_loss = np.mean(current_epoch_train_avg_loss_list)
                current_epoch_grads_avg_norms = (np.mean([elem[0] for elem in current_epoch_grads_norms_list]),
                                                 np.mean([elem[1] for elem in current_epoch_grads_norms_list]))

                current_epoch_grads_norms_list = []

                current_epoch_train_avg_loss_list = []
                current_epoch_train_avg_acc = np.mean(current_epoch_train_avg_acc_list)

                current_epoch_train_avg_acc_list = []
                current_epoch_val_avg_acc = np.mean(current_epoch_val_avg_acc_list)

                current_epoch_val_avg_acc_list = []
                args.lr *= args.lr_dec_rate

                if current_epoch_val_avg_acc > best_acc:
                    best_acc = current_epoch_val_avg_acc
                    best_loss = val_avg_loss
                    train_acc_of_best_model = current_epoch_train_avg_acc
                    # best_acc_score = val_avg_acc_score
                    # best_f1 = val_avg_f1
                    best_epoch = step
                    best_model.cpu()
                    del best_model
                    best_model = copy.deepcopy(net)

                # Monitor using Weights & Biases
                if args.wandb:
                    log2wandb(train_acc_of_best_model,
                              best_acc,
                              best_acc_score,
                              best_epoch,
                              best_f1,
                              best_loss,
                              step,
                              current_epoch_train_avg_loss,
                              current_epoch_train_avg_acc,
                              val_acc_dict,
                              val_acc_score_dict,
                              current_epoch_val_avg_acc,
                              val_avg_acc_score, val_avg_f1, val_avg_loss,
                              val_f1s_dict, val_loss_dict,
                              grads_norms=current_epoch_grads_avg_norms[0],
                              lr=args.lr,
                              clip=args.clip,
                              global_lr=global_lr)


    # # calibration
    # for j, c_id in enumerate(private_clients):
    #     calib_loader = val_loaders[c_id]
    #
    #     pbar_dict.update(
    #         {
    #             'Step': 'Cal',
    #             'Client': f'{c_id}'.zfill(3),
    #             'Client Number in Step': f'{(j + 1)}'.zfill(3),
    #             # 'Train Avg Loss': f'{train_avg_loss:.4f}',
    #             # 'Train Current Loss': f'{0.:.2f}'.zfill(5),
    #             # 'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
    #             # 'Val Avg Acc': f'{val_avg_acc:.4f}',
    #             # 'Best Avg Acc': f'{best_acc:.4f}'})
    #         })
    #     local_net, clib_avg_loss = local_train(args, net, calib_loader,
    #                                            pbar=step_iter, pbar_dict=pbar_dict)


    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders, plot_confusion_matrix=True)

    y_true_all, y_pred_all, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results
    # _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    if args.wandb:
        logtest2wandb(test_avg_acc)
    if args.wandb:
        wandb_plot_confusion_matrix(y_true_all, y_pred_all, list(range(args.num_classes)))

    # update_frame(args, dp_method='SGD_DP', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
    #              test_avg_acc=test_avg_acc)
