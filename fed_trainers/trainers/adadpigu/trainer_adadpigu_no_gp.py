import copy
import csv
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import trange

from adaclip import SparseAdaCliP, update_m_s
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from fed_trainers.trainers.adadpigu.utils import accumulate_importance, generate_topk_mask, MaskScheduler, save_args, \
    evaluate_on_trainset, save_times, save_summary
from fed_trainers.trainers.model import get_model
from fed_trainers.trainers.rdp_accountant import compute_rdp, get_privacy_spent
from fed_trainers.trainers.utils import get_sigma, get_clients, get_device, flatten_tensor, \
    load_aggregated_grads_to_global_net, eval_model, log2wandb, wandb_plot_confusion_matrix, update_frame, get_optimizer


def initialize_results_file(results_file, base_pruning_rate):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Base Pruning Rate: {base_pruning_rate}'])
        writer.writerow(
            ['Phase', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Epsilon Spent', 'Sigma'])


# def initialize_results_file(results_file, fieldnames, base_pruning_rate=None):
#     """
#     Initialize the results CSV file.
#     Optionally writes an experiment description/parameter line as a comment at the top.
#     Always writes fieldnames as the first line (for pandas compatibility).
#     """
#     if os.path.exists(results_file):
#         print(f"Warning: {results_file} already exists and will be overwritten.")
#     with open(results_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         if base_pruning_rate is not None:
#             writer.writerow([f'# Base Pruning Rate: {base_pruning_rate}'])
#         writer.writerow(fieldnames)


def log_results(results_file, phase, epoch, train_loss, train_acc, test_loss, test_acc, eps_spent=None, sigma=None):
    train_acc = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
    test_acc = test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc

    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if eps_spent is not None and sigma is not None:
            writer.writerow([phase, epoch, train_loss, train_acc, test_loss, test_acc, eps_spent, sigma])
        elif eps_spent is not None:
            writer.writerow([phase, epoch, train_loss, train_acc, test_loss, test_acc, eps_spent])
        else:
            writer.writerow([phase, epoch, train_loss, train_acc, test_loss, test_acc])


# def log_results(results_file, result_dict, fieldnames):

#     row = []
#     for k in fieldnames:
#         v = result_dict.get(k, '')
#         if isinstance(v, torch.Tensor):
#             v = v.item()
#         row.append(v)
#     with open(results_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(row)


def get_epsilon(q, steps, noise_multiplier, delta, orders=None):
    """
    Compute epsilon (privacy budget) using RDP accountant.
    Args:
        q (float): sampling rate per step (batch_size / n_total)
        steps (int): total number of steps
        noise_multiplier (float): noise multiplier (sigma)
        delta (float): target delta
        orders (np.ndarray or None): array of Renyi orders to use (default [1.1, 64.0, 0.1])
    Returns:
        eps (float): privacy budget epsilon
    """
    if orders is None:
        orders = np.arange(1.1, 64.0, 0.1)
    assert 0 < q < 1, "Sampling rate must be in (0,1)"
    assert steps > 0, "Number of steps must be positive"
    assert noise_multiplier > 0, "Noise multiplier must be positive"
    rdp = compute_rdp(q, noise_multiplier, steps, orders)
    eps, _, _ = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps

def local_train(
        model, train_loader, loss_fn, lr, epochs,
        dp=True, noise_multiplier=1.0, clip_norm=1.0,
        use_mask=False, pretrain_epochs=1, release_schedule=None, device=torch.device("cpu"),
        importance_scores=None, base_mask=None, adaclip=False,
        m_mat=None, s_mat=None, topk_ratio=0.1
):
    """
    Unified training function supporting:
    - Standard SGD
    - DP-SGD (vanilla or with mask)
    - Sparse AdaCliP (with/without mask)

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        loss_fn: loss function.
        lr: learning rate.
        epochs: total number of training epochs.
        dp: whether to enable DP-SGD.
        noise_multiplier: DP noise multiplier (sigma).
        clip_norm: DP gradient clipping norm.
        use_mask: whether to use structure mask.
        pretrain_epochs: epochs for importance score accumulation.
        release_schedule: list, release fractions for mask scheduler.
        device: torch.device for training.
        importance_scores: pre-computed importance scores (optional).
        base_mask: initial pruning mask (optional).
        adaclip: whether to use Sparse AdaCliP.
        m_mat, s_mat: AdaCliP moment vectors (for adaclip).
        topk_ratio: fraction of top-k gradients to update in SparseAdaCliP.

    Returns:
        model: trained model.
        final_mask: mask used in the last epoch (if any), otherwise None.
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    final_mask = None
    batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else None

    # ==== Prepare mask/importance if needed ====
    scheduler = None
    if dp or adaclip or use_mask:
        if importance_scores is None:
            importance_scores = accumulate_importance(
                model, train_loader, loss_fn, optimizer,
                pretrain_epochs, clip_norm, noise_multiplier, device
            )
        if base_mask is None:
            base_mask = generate_topk_mask(importance_scores, prune_fraction=0.1)
        if release_schedule is None or len(release_schedule) == 0:
            release_schedule = [1.0] * epochs  # Default: no gradual release
        scheduler = MaskScheduler(base_mask, importance_scores, release_schedule)

    # ==== Standard SGD ====
    if not dp:
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return model, final_mask

    # ==== Sparse AdaCliP ====
    if dp and adaclip:
        model.train()
        for epoch in range(epochs):
            current_mask = None
            if use_mask and scheduler is not None:
                current_mask = scheduler.step(epoch)
            recovered_grads = []

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                noisy_flat_grad, recovered_grad = SparseAdaCliP(
                    model, batch_x, batch_y, loss_fn,
                    m_vec=m_mat, s_vec=s_mat,
                    clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier,
                    topk_ratio=topk_ratio,
                    mask=current_mask
                )

                # Parameter update (manual)
                pointer = 0
                for param in model.parameters():
                    num_param = param.numel()
                    param_grad = noisy_flat_grad[pointer:pointer + num_param].view(param.shape)
                    param.data -= lr * param_grad
                    pointer += num_param

                # Collect mean recovered_grad for AdaCliP
                recovered_grads.append(recovered_grad.mean(dim=0))

            # Update m_mat, s_mat for AdaCliP
            if recovered_grads:
                recovered_grads = torch.stack(recovered_grads, dim=0)  # [num_batches, D]
                recovered_mean = recovered_grads.mean(dim=0)  # [D]
                m_mat, s_mat = update_m_s(recovered_mean, m_mat, s_mat, noise_multiplier)

        return model, current_mask

    # ==== Standard DP-SGD / Masked DP-SGD ====
    model.train()
    final_mask = None
    for epoch in range(epochs):
        current_mask = None
        if scheduler is not None:
            current_mask = scheduler.step(epoch)
        sum_clipped_grads = [torch.zeros_like(p) for p in model.parameters()]
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            B = batch_x.size(0)
            total_samples += B

            # Per-sample gradient DP-SGD
            for i in range(B):
                optimizer.zero_grad()
                output = model(batch_x[i].unsqueeze(0))
                loss = loss_fn(output, batch_y[i].unsqueeze(0))
                loss.backward()

                total_norm = torch.sqrt(sum(
                    p.grad.data.pow(2).sum() for p in model.parameters() if p.grad is not None
                ))
                factor = min(1.0, clip_norm / (total_norm + 1e-6))

                for idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        sum_clipped_grads[idx] += param.grad.data * factor

        # DP-SGD update (with noise, mask)
        sigma = noise_multiplier * clip_norm
        for idx, param in enumerate(model.parameters()):
            grad_avg = sum_clipped_grads[idx] / total_samples
            noise = torch.normal(0.0, sigma, size=param.shape, device=device)
            noisy_grad = grad_avg + noise / total_samples
            if current_mask is not None:
                noisy_grad = noisy_grad * current_mask[idx]
            param.data -= lr * noisy_grad

        final_mask = current_mask

    return model, final_mask


def apply_mask_to_param(model, mask):
    with torch.no_grad():
        for (name, param), m in zip(model.named_parameters(), mask):
            param.data *= m


def test(epoch, net, testloader, loss_func, use_cuda, best_acc, args, mask=None):
    """
    Evaluate the model on the test set. Optionally applies a static mask to parameters before evaluation
    to enforce sparsity structure (e.g., after pruning or sparse training).

    Args:
        epoch (int): Current epoch number.
        net: PyTorch model to evaluate.
        testloader: DataLoader for test data.
        loss_func: Loss function (should use reduction='mean').
        use_cuda (bool): Whether to use CUDA.
        best_acc (float): Current best test accuracy.
        args: Namespace, experiment arguments (for checkpoint saving).
        mask (list or None): Optional mask to statically apply to parameters before testing.

    Returns:
        test_loss (float): Average test loss.
        acc (float): Test set accuracy.
        best_acc (float): Updated best accuracy.
    """
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if mask is not None:
        backup_params = [p.data.clone() for p in net.parameters()]
        apply_mask_to_param(net, mask)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.data.max(1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)

    acc = 100. * correct / total
    print(f'Test Loss: {test_loss / (batch_idx + 1):.4f}, Test Acc: {acc:.2f}%')

    if mask is not None:
        # Restore original parameters after applying mask
        for p, b in zip(net.parameters(), backup_params):
            p.data.copy_(b)

    if acc > best_acc:
        print(f" New best model saved at epoch {epoch}: Test Acc = {acc:.2f}% (previous best {best_acc:.2f}%)")
        best_acc = acc
        # checkpoint(net, acc, epoch, args.sess)

    return test_loss / (batch_idx + 1), acc, best_acc


def compute_steps(epoch, batchsize, n_training):
    return int((epoch + 1) * n_training / batchsize)


def local_train_with_pruning(args, model, trainloader, pbar, pbar_dict):
    """
    Run the full three-stage training process with DP and structured pruning:
    1. Stage 1: DP pretraining and importance accumulation
    2. Stage 2: Progressive mask training (iterative structure release)
    3. Stage 3: Fixed mask finetuning

    Args:
        args: Namespace containing experiment arguments.
        model: PyTorch model to train.
        trainloader: DataLoader for training data.
        pbar: tqdm progress bar object.
        pbar_dict: Dictionary to store progress bar values.

    Returns:
        best_acc: Best test accuracy achieved during the process.
    """

    use_cuda = torch.cuda.is_available()
    device = get_device(cuda=use_cuda)
    local_net: torch.nn.Module = copy.deepcopy(model)
    local_net.train()
    optimizer = get_optimizer(args, local_net)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_start_time = time.time()
    times_list = []

    # === [Init] Load data and initialize model ===
    best_acc = 0

    # === Stage 1: DP Pretraining & Importance Accumulation ===
    print("==> Stage 1: Pretraining...")
    start_time = time.time()

    pretrain_epochs = 15

    importance_scores = [torch.zeros_like(p, device=device) for p in model.parameters()]
    lr = args.lr

    for epoch in range(pretrain_epochs):
        print(f"[Stage 1] Epoch {epoch + 1}/{pretrain_epochs}")
        local_net.train()
        for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)

            with backpack(BatchGrad()):
                loss.backward()

            B = batch_x.size(0)
            grad_norms = torch.zeros(B, device=device)
            for param in model.parameters():
                if not hasattr(param, "grad_batch"): continue
                grad_batch = param.grad_batch.reshape(B, -1)
                grad_norms += (grad_batch ** 2).sum(dim=1)
            grad_norms = grad_norms.sqrt()
            clip_factors = (args.clip / (grad_norms + 1e-6)).clamp(max=1.0)

            for idx, param in enumerate(model.parameters()):
                if not hasattr(param, "grad_batch"): continue
                grad_batch = param.grad_batch.reshape(B, -1)
                clipped = grad_batch * clip_factors.view(-1, 1)
                clipped_mean = clipped.mean(dim=0).view(param.shape)

                importance_scores[idx] += clipped_mean.abs()

                sigma = noise_multiplier * args.clip
                noise = torch.normal(0.0, sigma, size=param.shape, device=device)
                noisy_grad = clipped_mean + noise / B
                param.data -= lr * noisy_grad

        train_loss, train_acc = evaluate_on_trainset(model, trainloader, loss_fn, device)
        test_loss, test_acc, best_acc = test(epoch, model, testloader, loss_fn, use_cuda, best_acc, args, mask=None)
        eps_spent = get_epsilon(args.batchsize / args.n_training, (epoch + 1) * len(trainloader), noise_multiplier,
                                args.delta)
        # log_results(results_file, 'Stage 1 (Pretrain)', epoch + 1, train_loss, train_acc, test_loss, test_acc,
        #             eps_spent, sigma=noise_multiplier)

    elapsed = time.time() - start_time
    times_list.append(("Stage 1 (Pretrain)", elapsed))
    print(f" Stage 1 done in {elapsed:.2f} seconds.")

    # === Stage 2: Progressive Mask Training (iterative structure release) ===
    print("==> Stage 2: Progressive mask training...")
    print("==> Stage 2: Progressive mask training...")
    start_time = time.time()

    # Initialize AdaCliP vectors for coordinate-wise adaptive clipping
    total_dim = sum(p.numel() for p in model.parameters())
    m_mat = torch.zeros(total_dim, device=device)
    s_mat = torch.ones(total_dim, device=device)

    # Generate base pruning mask based on importance scores
    base_mask = generate_topk_mask(importance_scores, prune_fraction=base_pruning_rate)
    release_schedule = [i / 100 for i in range(2, 12, 2)]  # 2%, 4%, ..., 10%
    scheduler = MaskScheduler(base_mask, importance_scores, release_schedule)

    for epoch in range(len(release_schedule)):
        current_mask = scheduler.step(epoch)
        model, _ = local_train(
            model, trainloader, loss_fn, lr=args.lr, epochs=1,
            dp=True, noise_multiplier=noise_multiplier,
            clip_norm=args.clip, use_mask=True,
            pretrain_epochs=0,
            importance_scores=importance_scores,
            base_mask=base_mask,
            release_schedule=release_schedule,
            device=device,
            adaclip=True,
            m_mat=m_mat,
            s_mat=s_mat,
            topk_ratio=base_pruning_rate
        )

        train_loss, train_acc = evaluate_on_trainset(model, trainloader, loss_fn, device)
        print(f"[Stage 2][Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")

        test_loss, test_acc, best_acc = test(epoch, model, testloader, loss_fn, use_cuda, best_acc, args, current_mask)

        steps_so_far = compute_steps(pretrain_epochs + epoch, args.batchsize, args.n_training)
        eps_spent = get_epsilon(args.batchsize / args.n_training, steps_so_far, noise_multiplier, args.delta)
        log_results(results_file, 'Stage 2', epoch, train_loss, train_acc, test_loss, test_acc, eps_spent,
                    sigma=noise_multiplier)

    elapsed = time.time() - start_time
    times_list.append(("Stage 2 (Progressive Mask)", elapsed))
    print(f" Stage 2 done in {elapsed:.2f} seconds.")

    # === Stage 3: Fixed Mask Finetuning (sparse structure fixed) ===
    print("==> Stage 3: Fixed mask finetuning...")
    start_time = time.time()

    final_mask = scheduler.current_mask
    remaining_epochs = args.n_epoch - pretrain_epochs - len(release_schedule)

    for epoch in range(remaining_epochs):
        model, _ = local_train(
            model, trainloader, loss_fn, lr=args.lr, epochs=1,
            dp=True, noise_multiplier=noise_multiplier,
            clip_norm=args.clip, use_mask=True,
            pretrain_epochs=0,
            importance_scores=importance_scores,
            base_mask=base_mask,
            release_schedule=release_schedule,
            device=device,
            adaclip=True,
            m_mat=m_mat,
            s_mat=s_mat,
            topk_ratio=base_pruning_rate
        )

        train_loss, train_acc = evaluate_on_trainset(model, trainloader, loss_fn, device)
        print(f"[Stage 3][Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")

        test_loss, test_acc, best_acc = test(epoch, model, testloader, loss_fn, use_cuda, best_acc, args, final_mask)

        steps_so_far = compute_steps(pretrain_epochs + len(release_schedule) + epoch, args.batchsize, args.n_training)
        eps_spent = get_epsilon(args.batchsize / args.n_training, steps_so_far, noise_multiplier, args.delta)
        log_results(results_file, 'Stage 3', epoch, train_loss, train_acc, test_loss, test_acc, eps_spent,
                    sigma=noise_multiplier)

    elapsed = time.time() - start_time
    times_list.append(("Stage 3 (Fixed Mask)", elapsed))
    print(f" Stage 3 done in {elapsed:.2f} seconds.")

    # === Final Summary ===
    total_elapsed = time.time() - total_start_time
    print(f" Total training time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes).")

    save_times(times_list, results_file.replace('results.csv', 'times.csv'))
    save_summary(best_acc, eps_spent, total_elapsed, results_file.replace('results.csv', 'summary.txt'))

    return best_acc

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
