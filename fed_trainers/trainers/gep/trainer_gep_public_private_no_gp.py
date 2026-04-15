import copy
import logging
from collections import OrderedDict
from typing import Optional
import numpy as np
import torch
from tqdm import trange
from fed_trainers.trainers.gep.gep_utils import embed_grad, project_back_embedding, add_new_gradients_to_history, \
    compute_subspace
from fed_trainers.trainers.utils import get_device, local_train, flatten_tensor, eval_model, update_frame, \
    log2wandb, \
    load_aggregated_grads_to_global_net, compute_steps, compute_steps_in_epoch, logtest2wandb
from fed_trainers.trainers.factory import get_clients, get_model


def train(args, dataloaders):
    logger = logging.getLogger(args.log_name)

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1, train_acc_of_best_model = 0.0, 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    reconstruction_similarities = []
    public_clients, private_clients, dummy_clients = get_clients(args)
    num_public_clients = len(public_clients)
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
    # device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)

    basis_gradients: Optional[torch.Tensor] = None
    basis_gradients_cpu: Optional[torch.Tensor] = None

    train_loaders, val_loaders, test_loaders = dataloaders

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    reconstruction_similarity = 0.0
    num_steps = compute_steps(args)
    steps_in_epoch = compute_steps_in_epoch(args)
    current_epoch_val_avg_acc_list = []
    current_epoch_val_avg_acc = 0.0
    step_iter = trange(num_steps)

    pbar_dict = {'Step': '0', 'Client': '0', 'Public_Private?': 'Public_',
                 'Client Number in Step': '0', 'Best Epoch': '0', 'Val Avg Acc': '0.0',
                 'Best Avg Acc': '0.0', 'Train Avg Loss': '0.0'}
    for step in step_iter:

        # initialize global model params
        grads = OrderedDict()
        # public_params = OrderedDict()
        public_grads = OrderedDict()
        prev_params = OrderedDict()
        for n, p in net.named_parameters():
            grads[n] = []
            public_grads[n] = []
            prev_params[n] = p.detach()

        # *** Local trains on public clients - get gradients for subspace
        train_avg_loss_public, train_avg_acc_public = 0.0, 0.0
        for j, c_id in enumerate(public_clients):

            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3), 'Client': f'{c_id}'.zfill(3),
                              'Public_Private?': 'Public_',
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss_public:.4f}'})
            local_net, train_loss, train_acc = local_train(args, net, train_loader,
                                                    pbar=step_iter, pbar_dict=pbar_dict)

            train_avg_acc_public += (train_acc / num_public_clients)
            train_avg_loss_public += (train_loss / num_public_clients)

            # get client grads and sum.
            for n, p in local_net.named_parameters():
                public_grads[n].append(p.data.detach() - prev_params[n])

        public_grads_list = [torch.stack(public_grads[n]) for n, p in net.named_parameters()]

        public_grads_flat = flatten_tensor(public_grads_list)

        basis_gradients, basis_gradients_cpu, filled_history_size  = add_new_gradients_to_history(public_grads_flat, basis_gradients, basis_gradients_cpu, args.gradients_history_size)

        pca = compute_subspace(basis_gradients[:filled_history_size], args.basis_size, device)

        # *** End of public subspace update

        # Local trains on sampled clients

        # Sample several clients
        client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)

        train_avg_loss, train_avg_acc = 0.0, 0.0

        # Iterate over each client
        for j, c_id in enumerate(client_ids_step):

            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3), 'Client': f'{c_id}'.zfill(3),
                              'Public_Private?': 'Private',
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}',
                              'Train Current Loss': f'{0.:.4f}'.zfill(3),
                              'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                              'Reconstruction Similarity': f'{reconstruction_similarity:.4f}',
                              'Val Avg Acc': f'{val_avg_acc:.4f}',
                              'Best Avg Acc': f'{best_acc:.4f}'})

            local_net, train_loss, train_acc = local_train(args, net, train_loader,
                                                    pbar=step_iter, pbar_dict=pbar_dict)

            train_avg_acc += (train_acc / args.num_client_agg)
            train_avg_loss += (train_loss / args.num_client_agg)

            # get client grads
            for n, p in local_net.named_parameters():
                grads[n].append(p.data.detach() - prev_params[n])

        # stack sampled clients grads
        grads_list = [torch.stack(grads[n]) for n, p in net.named_parameters()]

        # flatten grads for clipping and noising
        grads_flattened = flatten_tensor(grads_list)

        # project grads to subspace computed by public grads
        embedded_grads = embed_grad(grads_flattened, pca).to(device)

        # clip grads in embedding  subspace
        embedded_grads_norms = torch.norm(embedded_grads, p=2, dim=-1)
        clip_factor = torch.max(torch.ones_like(embedded_grads_norms), embedded_grads_norms / args.clip)
        embedded_grads_clipped = torch.div(embedded_grads, clip_factor.reshape(-1, 1))

        # noise grads in embedding subspace
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip, size=embedded_grads_clipped.shape).to(
            device)
        noised_embedded_grads = embedded_grads_clipped + noise

        # aggregate sampled clients embedded grads and project back to gradient space
        reconstructed_grads = project_back_embedding(noised_embedded_grads, pca, device)

        # update subspace using private grads
        basis_gradients, basis_gradients_cpu, filled_history_size = add_new_gradients_to_history(reconstructed_grads,
                                                                                                 basis_gradients,
                                                                                                 basis_gradients_cpu,
                                                                                                 args.gradients_history_size)

        # reconstruction error
        norm_reconstructed = torch.norm(reconstructed_grads, p=2, dim=-1, keepdim=True)
        norm_original = torch.norm(grads_flattened, p=2, dim=-1, keepdim=True)
        similarity = (torch.linalg.vecdot(reconstructed_grads, grads_flattened, dim=-1).reshape(
            norm_reconstructed.shape) /
                      (norm_reconstructed * norm_original))

        reconstruction_similarity = float(torch.abs(similarity).mean())
        reconstruction_similarities.append(reconstruction_similarity)

        aggregated_grads = torch.mean(reconstructed_grads, dim=0)

        # update global net
        global_lr = args.global_lr ** (step // steps_in_epoch)
        logger.debug(f'Global learning rate: {global_lr}')
        net = load_aggregated_grads_to_global_net(aggregated_grads, net, prev_params, global_lr)

        # Evaluate model
        if ((step + 1) > args.eval_after and (step + 1) % args.eval_every == 0) or (step + 1) == num_steps:
            val_results = eval_model(args, net, private_clients, val_loaders)

            val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict, \
                val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results

            current_epoch_val_avg_acc_list.append(val_avg_acc)
            if len(current_epoch_val_avg_acc_list) >= (float(steps_in_epoch) / float(args.eval_every)):
                current_epoch_val_avg_acc = np.mean(current_epoch_val_avg_acc_list)
                current_epoch_val_avg_acc_list = []

                if current_epoch_val_avg_acc > best_acc:
                    best_acc = current_epoch_val_avg_acc
                    best_loss = val_avg_loss
                    train_acc_of_best_model = train_avg_acc
                    # best_acc_score = val_avg_acc_score
                    # best_f1 = val_avg_f1
                    best_epoch = step
                    best_model.cpu()
                    del best_model
                    best_model = copy.deepcopy(net)

        # Monitor using Weights & Biases
        if args.wandb:
            log2wandb(train_acc_of_best_model, best_acc, best_acc_score, best_epoch, best_f1, best_loss,
                      step,
                      train_avg_loss, train_avg_acc,
                      val_acc_dict,val_acc_score_dict,
                      current_epoch_val_avg_acc,
                      val_avg_acc_score, val_avg_f1, val_avg_loss,
                      val_f1s_dict, val_loss_dict)
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
    #             # 'Train Current Loss': f'{0.:.4f}'.zfill(3),
    #             # 'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
    #             # 'Val Avg Acc': f'{val_avg_acc:.4f}',
    #             # 'Best Avg Acc': f'{best_acc:.4f}'})
    #         })
    #     local_net, clib_avg_loss, clib_avg_acc = local_train(args, net, calib_loader,
    #                                            pbar=step_iter, pbar_dict=pbar_dict)
    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    if args.wandb:
        logtest2wandb(test_avg_acc)


    update_frame(args, dp_method='GEP_PUBLIC', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
                 test_avg_acc=test_avg_acc, reconstruction_similarity=np.mean(reconstruction_similarities))
