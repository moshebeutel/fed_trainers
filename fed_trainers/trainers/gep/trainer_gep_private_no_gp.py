import copy
import logging
from collections import OrderedDict
from typing import Optional, List
import numpy as np
import torch
from tqdm import trange
from gep_utils import add_new_gradients_to_history, compute_subspace, embed_grad, project_back_embedding
from fed_trainers.trainers.model import get_model
from fed_trainers.trainers.utils import get_clients, get_device, local_train, flatten_tensor, eval_model, update_frame, log2wandb, \
    load_aggregated_grads_to_global_net


def train(args, dataloaders):
    logger = logging.getLogger(args.log_name)

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1 = 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    reconstruction_similarities: List[float] = []
    public_clients, private_clients, dummy_clients = get_clients(args)
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)

    basis_gradients: Optional[torch.Tensor] = None
    basis_gradients_cpu: Optional[torch.Tensor] = None

    train_loaders, val_loaders, test_loaders = dataloaders

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    reconstruction_similarity = 0.0
    step_iter = trange(args.num_steps)

    pbar_dict = {'Step': '0', 'Client': '0',
                 'Client Number in Step': '0', 'Best Epoch': '0', 'Val Avg Acc': '0.0',
                 'Best Avg Acc': '0.0', 'Train Avg Loss': '0.0'}

    for step in step_iter:

        # select several clients
        client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)

        # initialize global model params
        grads = OrderedDict()
        prev_params = OrderedDict()
        for n, p in net.named_parameters():
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
                              'Reconstruction Similarity': f'{reconstruction_similarity:.4f}',
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
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip, size=grads_flattened_clipped.shape).to(
            device)
        noised_grads = grads_flattened_clipped + noise

        # update subspace using private grads
        basis_gradients, basis_gradients_cpu, filled_history_size  = add_new_gradients_to_history(noised_grads, basis_gradients, basis_gradients_cpu, args.gradients_history_size)

        pca = compute_subspace(basis_gradients[:filled_history_size], args.basis_size, device)

        # project grads to subspace
        embedded_grads = embed_grad(noised_grads, pca).to(device)
        # embedded_grads = embed_grad(grads_flattened, pca).to(device)

        # aggregate sampled clients grads and project back to gradient space
        reconstructed_grads = project_back_embedding(embedded_grads, pca, device)

        # reconstruction error
        norm_reconstructed = torch.norm(reconstructed_grads, p=2, dim=-1, keepdim=True)
        norm_original = torch.norm(grads_flattened, p=2, dim=-1, keepdim=True)
        similarity = (torch.linalg.vecdot(reconstructed_grads, grads_flattened, dim=-1).reshape(
            norm_reconstructed.shape) /
                      (norm_reconstructed * norm_original))

        reconstruction_similarity = float(torch.abs(similarity).mean())
        reconstruction_similarities.append(reconstruction_similarity)

        aggregated_grads = torch.mean(reconstructed_grads, dim=0)

        # update old parameters using private aggregated grads
        net = load_aggregated_grads_to_global_net(aggregated_grads, net, prev_params, args.global_lr)


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

        if args.wandb:
            log2wandb(best_acc, best_acc_score, best_epoch, best_f1, best_loss, step, train_avg_loss, val_acc_dict,
                      val_acc_score_dict, val_avg_acc, val_avg_acc_score, val_avg_f1, val_avg_loss, val_f1s_dict,
                      val_loss_dict)

    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    update_frame(args, dp_method='GEP_PRIVATE', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
                 test_avg_acc=test_avg_acc, reconstruction_similarity=np.median(reconstruction_similarities))
