import copy
import logging
from collections import OrderedDict
from typing import Optional, Dict
import numpy as np
import torch
from tqdm import trange
from gep_utils import embed_grad, project_back_embedding, add_new_gradients_to_history, \
    compute_subspace
from fed_trainers.trainers.model import get_model
from fed_trainers.trainers.utils import get_clients, get_device, local_train, flatten_tensor, eval_model, update_frame, log2wandb, \
    load_aggregated_grads_to_global_net, get_optimizer


def local_aux_train(args, net, train_loader, pbar, pbar_dict: Dict):
    local_net = copy.deepcopy(net)
    local_net.train()
    optimizer = get_optimizer(args, local_net)
    criteria = torch.nn.CrossEntropyLoss()
    device = get_device()
    num_channels = 16
    num_channels_aux = 24
    num_features_per_channel = 20
    num_classes = args.num_classes
    train_avg_loss = 0.0
    for k, batch in enumerate(train_loader):
        x, Y = batch
        sampled_channels = np.random.choice(range(num_channels_aux), size=num_channels, replace=False)
        x = x.reshape(-1, num_channels_aux, num_features_per_channel)
        x = x[:, sampled_channels, :]
        x = x.reshape(-1, num_channels * num_features_per_channel)
        Y = torch.from_numpy(np.random.choice(list(range(num_classes)), size=Y.size(), replace=True))
        x = x.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        # forward prop
        pred = local_net(x)
        loss = criteria(pred, Y)

        # back prop
        loss.backward()
        # # clip gradients
        # torch.nn.utils.clip_grad_norm_(local_net.parameters(), args.clip)
        # update local parameters
        optimizer.step()

        # aggregate losses
        train_avg_loss += (loss.item() / Y.shape[0])

        pbar_dict.update({"Inner Step": f'1'.zfill(3),
                          "Batch": f'{(k + 1)}'.zfill(3),
                          "Train Current Loss": f'{loss.item():5.2f}'})
        pbar.set_postfix(pbar_dict)

    # end of for k, batch in enumerate(train_loader):
    return local_net, train_avg_loss


def train(args, dataloaders):
    logger = logging.getLogger(args.log_name)

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1 = 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    reconstruction_similarities = []
    public_clients, private_clients, dummy_clients = get_clients(args)
    device = get_device()
    # device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)

    basis_gradients: Optional[torch.Tensor] = None
    basis_gradients_cpu: Optional[torch.Tensor] = None

    train_loaders, val_loaders, test_loaders, aux_loaders = dataloaders
    aux_clients = aux_loaders.keys()

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    reconstruction_similarity = 0.0
    reconstruction_error = 0.0
    step_iter = trange(args.num_steps)

    pbar_dict = {'Step': '0', 'Client': '0', 'Auxiliary_Private?': 'Auxiliary',
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

        # *** Local trains on auxiliary public datasets - get gradients for subspace
        train_avg_loss = 0.0
        for j, c_id in enumerate(aux_clients):

            train_loader = aux_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3), 'Client': f'{c_id}'.zfill(3),
                              'Auxiliary_Private?': 'Auxiliary',
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}'})
            local_net, train_avg_loss = local_aux_train(args, net, train_loader,
                                                        pbar=step_iter, pbar_dict=pbar_dict)

            # get client grads and sum.
            for n, p in local_net.named_parameters():
                public_grads[n].append(p.data.detach() - prev_params[n])

        public_grads_list = [torch.stack(public_grads[n]) for n, p in net.named_parameters()]

        public_grads_flat = flatten_tensor(public_grads_list)

        basis_gradients, basis_gradients_cpu, filled_history_size = add_new_gradients_to_history(public_grads_flat,
                                                                                                 basis_gradients,
                                                                                                 basis_gradients_cpu,
                                                                                                 args.gradients_history_size)

        pca = compute_subspace(basis_gradients[:filled_history_size], args.basis_size, device)

        # logger.debug(f'#$% step: {step}')
        # logger.debug(f'#$% basis size: {args.basis_size} history {args.gradients_history_size} lr {args.lr} clip {args.clip}')
        # logger.debug(f'#$% ********************************************************************')
        # logger.debug(f'#$% translate transform: {torch.abs(pca[1]).mean().item()}')
        # logger.debug(f'#$% scale transform: {pca[2]}')
        # logger.debug(f'#$% explained_variance_ratio_: {pca[3].squeeze().tolist()}')
        # logger.debug(f'#$% explained_variance_ratio_cumsum: {pca[4].squeeze().tolist()}')

        # *** End of public subspace update

        # Local trains on sampled clients

        # Sample several clients
        # client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)
        client_ids_step = np.random.choice([*public_clients, *private_clients], size=args.num_client_agg, replace=False)

        # Iterate over each client
        train_avg_loss = 0
        for j, c_id in enumerate(client_ids_step):

            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3), 'Client': f'{c_id}'.zfill(3),
                              'Auxiliary_Private?': 'Private'.center(9),
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}',
                              'Train Current Loss': f'{0.:.4f}',
                              'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                              'Reconstruction Similarity': f'{reconstruction_similarity:.4f}',
                              'Reconstruction Error': f'{reconstruction_error:.4f}',
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
        reconstruction_error = torch.dist(reconstructed_grads, grads_flattened)

        # reconstruction error
        norm_reconstructed = torch.norm(reconstructed_grads, p=2, dim=-1, keepdim=True)
        norm_original = torch.norm(grads_flattened, p=2, dim=-1, keepdim=True)
        similarity = (torch.linalg.vecdot(reconstructed_grads, grads_flattened, dim=-1).reshape(
            norm_reconstructed.shape) /
                      (norm_reconstructed * norm_original))

        reconstruction_similarity = float(torch.abs(similarity).mean())
        reconstruction_similarities.append(reconstruction_similarity)

        # logger.debug(f'#$% norm_reconstructed: {norm_reconstructed.tolist()}')
        # logger.debug(f'#$% norm_original: {norm_original.tolist()}')
        logger.debug(f'#$% norm ratio norm: {float(torch.abs(norm_reconstructed / norm_original).mean()):.4f}')
        logger.debug(f'#$% reconstruction_similarity: {reconstruction_similarity:.4f}')

        aggregated_grads = torch.mean(reconstructed_grads, dim=0)

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

    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    update_frame(args, dp_method='GEP_PUBLIC', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
                 test_avg_acc=test_avg_acc, reconstruction_similarity=np.mean(reconstruction_similarities))
