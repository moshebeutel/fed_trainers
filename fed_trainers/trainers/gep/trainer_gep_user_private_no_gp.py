import copy
import gc
import logging
from collections import OrderedDict
from typing import Optional, List, Dict
import numpy as np
import torch
from backpack.extensions import BatchGrad
from tqdm import trange
from fed_trainers.trainers.gep.gep_utils import add_new_gradients_to_history, compute_subspace, embed_grad, project_back_embedding
from fed_trainers.trainers.model import get_model
from fed_trainers.trainers.utils import get_clients, get_device, local_train, flatten_tensor, eval_model, update_frame, \
    log2wandb, \
    load_aggregated_grads_to_global_net, compute_steps, get_sigma, get_optimizer
from backpack import extend, backpack

def local_user_private_train(args, net, train_loader, pbar, pbar_dict: Dict):
    local_net = copy.deepcopy(net)
    optimizer = get_optimizer(args, local_net)
    criteria = torch.nn.CrossEntropyLoss()
    local_net = extend(local_net)
    criteria = extend(criteria)
    device = get_device()
    train_avg_loss = 0.0
    grads = OrderedDict()
    for n, p in net.named_parameters():
        grads[n] = []

    local_net.train()
    for i in range(args.inner_steps):
        for k, batch in enumerate(train_loader):
            x, Y = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # forward prop
            pred = local_net(x)
            loss = criteria(pred, Y)

            # back prop
            with backpack(BatchGrad()):
                loss.backward()
            for n,p in local_net.named_parameters():
                assert hasattr(p, "grad_batch"), f"Parameter {p} does not have grad_batch attribute"
                assert p.grad_batch is not None, f"Parameter {p} has None grad_batch"
                assert isinstance(p.grad_batch, torch.Tensor), f"Parameter {p} has incorrect type for grad_batch"
                assert p.grad_batch.shape[0] == Y.shape[0], f"Parameter {p} has incorrect batch size"
                assert p.grad_batch.shape[1:] == p.shape, f"Parameter {p} has incorrect shape"
                grads[n].append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                p.grad_batch = p.grad_batch.detach().cpu()
                p.grad_batch = None
                del p.grad_batch

            # update local parameters
            optimizer.step()

            # aggregate losses
            train_avg_loss += (loss.item() / Y.shape[0])

            pbar_dict.update({"Inner Step": f'{(i + 1)}'.zfill(3),
                              "Batch": f'{(k + 1)}'.zfill(3),
                              "Train Current Loss": f'{loss.item():5.2f}'.zfill(3)})
            pbar.set_postfix(pbar_dict)

            pred = pred.detach().cpu()
            x = x.detach().cpu()
            Y = Y.detach().cpu()
            x, Y, pred = None, None, None
            del pred, Y, x

        # end of for k, batch in enumerate(train_loader):
    # end of for i in range(args.inner_steps):

    # stack sampled clients grads
    grads_list = [torch.cat(grads[n]) for n, p in net.named_parameters()]

    flat_grad_batch_tensor = flatten_tensor(grads_list)

    # free memory

    local_net.to('cpu')
    local_net = None
    del local_net
    optimizer = None
    del optimizer
    grads_list = [t.detach().cpu() for t in grads_list]
    grads_list = None
    del grads_list
    for n, p in net.named_parameters():
        grads[n] = [t.detach().cpu() for t in grads[n]]
        grads[n] = None
    grads=None
    del grads

    gc.collect()
    torch.cuda.empty_cache()

    return train_avg_loss, flat_grad_batch_tensor

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

    # basis_gradients = {private_client: None for private_client in private_clients}
    # basis_gradients_cpu = {private_client: None for private_client in private_clients}

    train_loaders, val_loaders, test_loaders = dataloaders

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    reconstruction_similarity = 0.0
    num_steps = compute_steps(args)
    step_iter = trange(num_steps)

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
        reconstructed_grads_list = []
        logger.debug(f'Client ids in step {step}: {client_ids_step}')

        # iterate over each client
        train_avg_loss = 0
        for j, c_id in enumerate(client_ids_step):

            train_loader = train_loaders[c_id]

            pbar_dict.update({'Step': f'{(step + 1)}'.zfill(3),
                              'Client': f'{c_id}'.zfill(3),
                              'Client Number in Step': f'{(j + 1)}'.zfill(3),
                              'Train Avg Loss': f'{train_avg_loss:.4f}',
                              'Train Current Loss': f'{0.:.4f}'.zfill(3),
                              'Best Epoch': f'{(best_epoch + 1)}'.zfill(3),
                              'Reconstruction Similarity': f'{reconstruction_similarity:.4f}',
                              'Val Avg Acc': f'{val_avg_acc:.4f}',
                              'Best Avg Acc': f'{best_acc:.4f}'})

            train_avg_loss, flat_grad_user_tensor = local_user_private_train(args, net, train_loader,
                                                    pbar=step_iter, pbar_dict=pbar_dict)


            # clip grads
            grads_norms = torch.norm(flat_grad_user_tensor, p=2, dim=-1)
            clip_factor = torch.max(torch.ones_like(grads_norms), grads_norms / args.clip)
            grads_flattened_clipped = torch.div(flat_grad_user_tensor, clip_factor.reshape(-1, 1))

            # update subspace using private grads
            # basis_gradients[c_id], basis_gradients_cpu[c_id], filled_history_size = add_new_gradients_to_history(noised_grads, basis_gradients[c_id], basis_gradients_cpu[c_id], args.gradients_history_size)

            pca = compute_subspace(grads_flattened_clipped, args.basis_size, device)

            # project grads to subspace
            embedded_grads = embed_grad(grads_flattened_clipped, pca).to(device)
            # embedded_grads = embed_grad(grads_flattened, pca).to(device)

            # noise grads
            noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                                 size=embedded_grads.shape).to(device)
            noised_grads = embedded_grads + noise

            # aggregate sampled clients grads and project back to gradient space
            reconstructed_grads = project_back_embedding(noised_grads, pca, device)
            reconstructed_grads_list.append(reconstructed_grads)


            # free memory
            flat_grad_user_tensor = flat_grad_user_tensor.detach().cpu()
            grads_flattened_clipped = grads_flattened_clipped.detach().cpu()
            noised_grads = noised_grads.detach().cpu()
            embedded_grads = embedded_grads.detach().cpu()
            reconstructed_grads = reconstructed_grads.detach().cpu()

            flat_grad_user_tensor = None
            grads_flattened_clipped = None
            noised_grads = None
            embedded_grads = None
            del flat_grad_user_tensor, grads_flattened_clipped, noised_grads, embedded_grads, reconstructed_grads
            gc.collect()
            torch.cuda.empty_cache()

        # # reconstruction error
        # norm_reconstructed = torch.norm(reconstructed_grads, p=2, dim=-1, keepdim=True)
        # norm_original = torch.norm(grads_flattened, p=2, dim=-1, keepdim=True)
        # similarity = (torch.linalg.vecdot(reconstructed_grads, grads_flattened, dim=-1).reshape(
        #     norm_reconstructed.shape) /
        #               (norm_reconstructed * norm_original))
        #
        # reconstruction_similarity = float(torch.abs(similarity).mean())
        # reconstruction_similarities.append(reconstruction_similarity)
        reconstructed_grads_tensor = torch.cat(reconstructed_grads_list, dim=0)
        aggregated_grads = torch.mean(reconstructed_grads_tensor, dim=0)

        # update global net
        global_lr = args.global_lr ** step
        logger.debug(f'Global learning rate: {global_lr}')
        net = load_aggregated_grads_to_global_net(aggregated_grads, net, prev_params, args.global_lr)

        # free memory
        reconstructed_grads_list = [None for _ in reconstructed_grads_list]
        reconstructed_grads_list = None
        del reconstructed_grads_list
        reconstructed_grads_tensor = None
        del reconstructed_grads_tensor
        aggregated_grads = None
        del aggregated_grads
        gc.collect()
        torch.cuda.empty_cache()



        if ((step + 1) > args.eval_after and (step + 1) % args.eval_every == 0) or (step + 1) == num_steps:
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
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    y_true_all, y_pred_all, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results
    # _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logger.info(f'## Test Results For Args {args}: test acc {test_avg_acc:.4f}, test loss {test_avg_loss:.4f} ##')

    # if args.wandb:
    #     wandb_plot_confusion_matrix(y_true_all, y_pred_all, list(range(args.num_classes)))


    update_frame(args, dp_method='GEP_PRIVATE', epoch_of_best_val=best_epoch, best_val_acc=best_acc,
                 test_avg_acc=test_avg_acc, reconstruction_similarity=np.median(reconstruction_similarities))
