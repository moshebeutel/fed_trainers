import argparse
import csv
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from adaclip import SparseAdaCliP, update_m_s
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from models.get_model import get_model
from rdp_accountant import compute_rdp, get_privacy_spent
from utils import get_data_loader, get_sigma

parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=2, type=int, help='random seed')

parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
parser.add_argument('--n_epoch', default=500, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

parser.add_argument('--private', action='store_true', help='enable differential privacy')
parser.add_argument('--clip', default=5., type=float, help='gradient clipping bound')
parser.add_argument('--eps', default=4., type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

args = parser.parse_args()
assert args.dataset in ['cifar10', 'svhn', 'mnist', 'fmnist']


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


use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0
batch_size = args.batchsize

if args.seed != -1:
    seed_value = int(time.time())
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

# Prepare data loaders
trainloader, testloader, n_training, n_test = get_data_loader(args.dataset, batchsize=args.batchsize)
print(f'# of training examples: {n_training}, # of testing examples: {n_test}')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.seed is not None and args.seed >= 0:
    print(f'[Info] Using fixed seed: {args.seed}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
else:
    seed_value = int(time.time())
    print(f'[Warning] No seed provided, using {seed_value} for randomness.')
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)

trainloader, testloader, n_training, n_test = get_data_loader(args.dataset, batchsize=args.batchsize)
print(f'# of training examples: {n_training}, # of testing examples: {n_test}')


# Model initialization
def initialize_net(args):
    net = get_model(algorithm='DPSGD' if args.private else 'NonDP', dataset_name=args.dataset.upper(), device='cuda')
    net = extend(net)
    return net


if args.resume:
    try:
        checkpoint_path = f'./checkpoint/{args.sess}.ckpt'
        checkpoint_data = torch.load(checkpoint_path)
        net = initialize_net(args)
        net.load_state_dict(checkpoint_data['net'])
        best_acc = checkpoint_data['acc']
        start_epoch = checkpoint_data['epoch'] + 1
        torch.set_rng_state(checkpoint_data['rng_state'])
        print('Checkpoint successfully restored.')
    except Exception as e:
        print(f'Error during checkpoint restoration: {e}')
        net = initialize_net(args)
        best_acc, start_epoch = 0, 0
else:
    net = initialize_net(args)
    best_acc, start_epoch = 0, 0

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {num_params / 1e6:.3f} M')

print(f'\n==> Computing noise scale for privacy budget ({args.eps:.2f}, {args.delta})-DP')
q = args.batchsize / n_training
steps = int(args.n_epoch * n_training / args.batchsize)
noise_multiplier, real_eps = get_sigma(q, steps, args.eps, args.delta, rgp=False)
print(f'Sampling probability (q): {q:.5f}')
print(f'Total steps: {steps}')
print(f'Noise multiplier (sigma): {noise_multiplier:.4f}')
print(f'Estimated privacy spent (epsilon): {real_eps:.4f}')


def initialize_model(args):
    net = initialize_net(args)
    initial_state_dict = {name: param.clone() for name, param in net.named_parameters()}

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    loss_func = nn.CrossEntropyLoss(reduction='sum' if args.private else 'mean')
    loss_func = extend(loss_func)

    return net, initial_state_dict, optimizer, loss_func


# def initialize_model(args, device):

#     net = initialize_net(args, device=device)
#     initial_state_dict = {name: param.clone() for name, param in net.named_parameters()}

#     optimizer = optim.SGD(
#         net.parameters(),
#         lr=args.lr,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay
#     )


# reduction_mode = 'sum' if args.private else 'mean'
# loss_func = nn.CrossEntropyLoss(reduction=reduction_mode)
# loss_func = extend(loss_func)

# return net, optimizer, loss_func, initial_state_dict

def get_epsilon(sampling_rate, steps, noise_multiplier, delta):
    orders = np.arange(1.1, 64.0, 0.1)
    rdp = compute_rdp(sampling_rate, noise_multiplier, steps, orders)
    eps, _, _ = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps


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


def save_args(args, save_path):
    with open(save_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


def save_args(args, save_path, fmt='txt'):
    """
    Save experiment arguments to file (txt/json).
    Args:
        args: argparse.Namespace object
        save_path: file path
        fmt: 'txt' or 'json'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    args_dict = vars(args)
    if fmt == 'json':
        with open(save_path, 'w') as f:
            json.dump(args_dict, f, indent=2)
    else:
        with open(save_path, 'w') as f:
            for k in sorted(args_dict.keys()):
                f.write(f"{k}: {args_dict[k]}\n")


def save_times(times_list, save_path):
    with open(save_path, mode='w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Time (seconds)', 'Time (minutes)'])
        for name, sec in times_list:
            writer.writerow([name, f"{sec:.2f}", f"{sec / 60:.2f}"])


def save_times(times_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Time (seconds)', 'Time (minutes)'])
        for name, sec in times_list:
            try:
                sec = float(sec)
            except Exception:
                sec = 0.0
            writer.writerow([name, f"{sec:.2f}", f"{sec / 60:.2f}"])


def save_summary(test_acc, eps_used, total_elapsed, save_path):
    with open(save_path, mode='w') as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Final Epsilon: {eps_used:.4f}\n")
        f.write(f"Total Training Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)\n")


def save_summary(test_acc, eps_used, total_elapsed, save_path):
    """
    Save final experiment summary to txt.
    Args:
        test_acc (float): Final test accuracy, 0-100 scale
        eps_used (float): Final epsilon used
        total_elapsed (float): Total elapsed time, seconds
        save_path (str): Save file path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w') as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Final Epsilon: {eps_used:.4f}\n")
        f.write(f"Total Training Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)\n")
    return save_path


class MaskScheduler:
    def __init__(self, base_mask, importance_scores, release_schedule):
        """
        Mask gradual-release scheduler for structured sparse training.

        Args:
            base_mask (list of Tensor): List matching the model parameter structure.
                base_mask[i] is the initial mask tensor (0 or 1) for each parameter group.
                Elements with value 1 are initially trainable; 0 means pruned/frozen.
            importance_scores (list of Tensor): List of accumulated importance scores
                (same structure as base_mask). Used to determine the order of release—
                parameters with higher importance are released earlier.
            release_schedule (list of float): Each element is a cumulative ratio in [0, 1],
                specifying the total proportion of trainable parameters to be unfrozen at each epoch.
                For example, [0.02, 0.04, 0.06, 0.08, 0.10] means gradually releasing up to 10%.
        """
        assert len(base_mask) == len(importance_scores), "base_mask and importance_scores must have the same length."
        self.base_mask = [m.clone() for m in base_mask]  # Deep copy for safety
        self.release_schedule = list(release_schedule)  # Copy the schedule for internal use

        # Ensure the release_schedule is monotonically non-decreasing
        for i in range(1, len(self.release_schedule)):
            if self.release_schedule[i] < self.release_schedule[i - 1]:
                self.release_schedule[i] = self.release_schedule[i - 1]

        # Collect global indices and importance scores for all initially trainable parameters (mask==1)
        flat_scores = []
        flat_indices = []
        offset = 0
        for mask, scores in zip(self.base_mask, importance_scores):
            mask_flat = mask.flatten()
            scores_flat = scores.flatten()
            numel = mask_flat.numel()
            for j in range(numel):
                if mask_flat[j].item() == 1:
                    flat_scores.append(scores_flat[j].item())
                    flat_indices.append(offset + j)
            offset += numel

        if len(flat_scores) == 0:
            # Edge case: no parameters available for release
            self.sorted_release_indices = torch.tensor([], dtype=torch.long)
        else:
            flat_scores = torch.tensor(flat_scores)
            flat_indices = torch.tensor(flat_indices, dtype=torch.long)
            sorted_idx = torch.argsort(flat_scores, descending=True)
            # List of global parameter indices (trainable subset), sorted by importance descending
            self.sorted_release_indices = flat_indices[sorted_idx]

        self.total_release_count = self.sorted_release_indices.numel()
        self.current_mask = [m.clone() for m in self.base_mask]  # Current mask (deep copy)
        self.current_epoch = -1  # -1 indicates scheduler not yet stepped

    def step(self, epoch):
        self.current_epoch = epoch
        if self.total_release_count == 0:
            return self.current_mask
        if epoch < len(self.release_schedule):
            target_fraction = self.release_schedule[epoch]
        else:
            target_fraction = self.release_schedule[-1]
        release_count = int(self.total_release_count * target_fraction)
        release_count = min(release_count, self.total_release_count)
        flat_mask = torch.cat([m.flatten() for m in self.base_mask]).clone()
        if release_count > 0:
            indices_to_unmask = self.sorted_release_indices[:release_count]
            flat_mask[indices_to_unmask] = 1
        new_mask_list = []
        offset = 0
        for mask in self.base_mask:
            numel = mask.numel()
            new_mask = flat_mask[offset: offset + numel].view(mask.shape)
            new_mask_list.append(new_mask)
            offset += numel
        self.current_mask = new_mask_list
        return self.current_mask

    def step(self, epoch):
        """
        Update mask according to the release schedule at the given epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            list of Tensor: The current mask list (structure matches model parameters).
        """
        self.current_epoch = epoch
        if self.total_release_count == 0:
            # No parameters available for release; return the current mask
            return [m.clone() for m in self.current_mask]
        # Determine cumulative release fraction at this epoch
        if epoch < len(self.release_schedule):
            target_fraction = self.release_schedule[epoch]
        else:
            target_fraction = self.release_schedule[-1]
        # Compute the number of parameters to unmask (cumulative)
        release_count = int(self.total_release_count * target_fraction)
        release_count = min(release_count, self.total_release_count)
        # Construct new mask: start from base_mask, set the top-k important parameters to 1
        flat_mask = torch.cat([m.flatten() for m in self.base_mask]).clone()
        if release_count > 0:
            indices_to_unmask = self.sorted_release_indices[:release_count]
            flat_mask[indices_to_unmask] = 1  # Unmask these parameters (set to 1)
        # Reshape flat_mask back to the original parameter structure
        new_mask_list = []
        offset = 0
        for mask in self.base_mask:
            numel = mask.numel()
            new_mask = flat_mask[offset: offset + numel].view(mask.shape)
            new_mask_list.append(new_mask)
            offset += numel
        self.current_mask = new_mask_list
        return [m.clone() for m in self.current_mask]


def generate_topk_mask(importance_scores, prune_fraction=0.1):
    """
    Generate a Top-k global pruning mask based on importance scores.
    This sets the lowest `prune_fraction` proportion of importance scores to zero (pruned), and keeps the rest as one.

    Args:
        importance_scores: List of tensors, each with the same shape as a model parameter tensor,
            containing the accumulated importance score (e.g., gradient magnitude) for each parameter.
        prune_fraction: Float in (0, 1), fraction of total parameters to prune (e.g., 0.1 means prune 10% with lowest scores).

    Returns:
        mask_list: List of 0/1 tensors, same structure as importance_scores.
            Each mask[i][j]=1 means keep this parameter, mask[i][j]=0 means prune/freeze.
    """
    # Concatenate all importance scores into a single flat tensor
    flat_scores = torch.cat([scores.flatten() for scores in importance_scores])
    total_params = flat_scores.numel()
    k = int(total_params * prune_fraction)
    if k <= 0:
        # No pruning if k <= 0: keep all parameters
        return [torch.ones_like(scores) for scores in importance_scores]
    # Find the threshold: the k-th smallest score across all parameters
    topk_values, topk_indices = torch.topk(flat_scores, k, largest=False)
    threshold = topk_values.max()  # Prune all parameters <= this score

    # Generate mask for each parameter tensor
    mask_list = []
    for scores in importance_scores:
        mask = torch.ones_like(scores)
        mask[scores <= threshold] = 0
        mask_list.append(mask)
    return mask_list


def generate_topk_mask(importance_scores, prune_fraction=0.1):
    """
    Generate a global Top-k pruning mask based on importance scores.
    Args:
        importance_scores (list of Tensor): List matching model.parameters(), each tensor is the accumulated importance for that parameter.
        prune_fraction (float): Fraction (0~1) of least important parameters to prune (set mask to 0).
    Returns:
        mask_list (list of Tensor): List matching model.parameters(), each element is a 0/1 mask tensor (1: keep, 0: prune/freeze).
    """
    # Flatten and concatenate all importance scores into one long tensor
    flat_scores = torch.cat([scores.flatten() for scores in importance_scores])
    total_params = flat_scores.numel()
    k = int(total_params * prune_fraction)
    if k <= 0:
        # No pruning: return all-ones mask
        return [torch.ones_like(scores) for scores in importance_scores]
    # Find the global threshold: the k-th smallest importance value
    topk_values, _ = torch.topk(flat_scores, k, largest=False)
    threshold = topk_values.max()
    # Generate mask: 1 for parameters above threshold, 0 for those below or equal to threshold
    mask_list = []
    for scores in importance_scores:
        mask = torch.ones_like(scores)
        mask[scores <= threshold] = 0
        mask_list.append(mask)
    return mask_list


def evaluate_on_trainset(model, train_loader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc


def evaluate_on_trainset(model, train_loader, loss_fn, device):
    """
    Evaluate model performance on the training set.

    Args:
        model: PyTorch model to evaluate
        train_loader: DataLoader for training data
        loss_fn: loss function (should have reduction='mean' for correct averaging)
        device: torch.device

    Returns:
        avg_loss (float): average loss per batch or per sample (depending on loss_fn)
        acc (float): accuracy on the training set
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    if total == 0:
        return 0.0, 0.0  # Or raise an exception/warning
    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc


def accumulate_importance(model, train_loader, loss_fn, optimizer, pretrain_epochs,
                          clip_norm, noise_multiplier, device=torch.device("cpu")):
    model.to(device)
    model.train()
    importance_scores = [torch.zeros_like(p, device=device) for p in model.parameters()]
    lr = optimizer.param_groups[0]['lr']

    for epoch in range(pretrain_epochs):
        print(f'[Stage 1] Epoch {epoch + 1}/{pretrain_epochs}')
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
            clip_factors = (clip_norm / (grad_norms + 1e-6)).clamp(max=1.0)

            for idx, param in enumerate(model.parameters()):
                if not hasattr(param, "grad_batch"): continue
                grad_batch = param.grad_batch.reshape(B, -1)
                clipped = grad_batch * clip_factors.view(-1, 1)
                clipped_mean = clipped.mean(dim=0).view(param.shape)

                importance_scores[idx] += clipped_mean.abs()

                sigma = noise_multiplier * clip_norm
                noise = torch.normal(0.0, sigma, size=param.shape, device=device)
                noisy_grad = clipped_mean + noise / B
                param.data -= lr * noisy_grad

        train_loss, train_acc = evaluate_on_trainset(model, train_loader, loss_fn, device)
        print(f"    [Eval] Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")

    return importance_scores


def accumulate_importance(
        model, train_loader, loss_fn, optimizer, pretrain_epochs,
        clip_norm, noise_multiplier, device=torch.device("cpu")
):
    """
    DP-SGD pretraining with BackPACK: accumulate parameter-wise importance scores (sum of abs(clipped mean gradients)).
    Evaluates train loss/accuracy each epoch.

    Args:
        model: PyTorch model (should be extended by BackPACK).
        train_loader: DataLoader for pretraining data.
        loss_fn: loss function (should be extended by BackPACK).
        optimizer: SGD optimizer (used only for lr retrieval).
        pretrain_epochs: number of pretraining epochs.
        clip_norm: DP-SGD clipping norm.
        noise_multiplier: DP-SGD noise multiplier (sigma).
        device: torch.device.

    Returns:
        importance_scores (list of Tensor): same structure as model.parameters(), cumulative importance.
    """
    model.to(device)
    model.train()
    importance_scores = [torch.zeros_like(p, device=device) for p in model.parameters()]
    lr = optimizer.param_groups[0]['lr']

    for epoch in range(pretrain_epochs):
        print(f'[Stage 1] Epoch {epoch + 1}/{pretrain_epochs}')
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
            clip_factors = (clip_norm / (grad_norms + 1e-6)).clamp(max=1.0)

            for idx, param in enumerate(model.parameters()):
                if not hasattr(param, "grad_batch"): continue
                grad_batch = param.grad_batch.reshape(B, -1)
                clipped = grad_batch * clip_factors.view(-1, 1)
                clipped_mean = clipped.mean(dim=0).view(param.shape)

                # Accumulate importance: sum of abs(clipped mean gradient)
                importance_scores[idx] += clipped_mean.abs()

                # Parameter update: add Gaussian noise (DP-SGD)
                sigma = noise_multiplier * clip_norm
                noise = torch.normal(0.0, sigma, size=param.shape, device=device)
                noisy_grad = clipped_mean + noise / B
                param.data -= lr * noisy_grad

        # Training set evaluation after each epoch
        train_loss, train_acc = evaluate_on_trainset(model, train_loader, loss_fn, device)
        print(f"    [Eval] Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")

    return importance_scores


def train(
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
            correct += predicted.eq(targets.data).cpu().sum()
            total += targets.size(0)

    acc = 100. * correct / total
    print(f'Test Loss: {test_loss / (batch_idx + 1):.4f}, Test Acc: {acc:.2f}%')

    if mask is not None:
        for p, b in zip(net.parameters(), backup_params):
            p.data.copy_(b)

    if acc > best_acc:
        print(f" New best model saved at epoch {epoch}: Test Acc = {acc:.2f}% (previous best {best_acc:.2f}%)")
        best_acc = acc
        checkpoint(net, acc, epoch, args.sess)

    return test_loss / (batch_idx + 1), acc, best_acc


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


def train_with_pruning(args, results_file, base_pruning_rate):
    """
    Run the full three-stage training process with DP and structured pruning:
    1. Stage 1: DP pretraining and importance accumulation
    2. Stage 2: Progressive mask training (iterative structure release)
    3. Stage 3: Fixed mask finetuning

    Args:
        args: Namespace containing experiment arguments.
        results_file: Path to save results CSV file.
        base_pruning_rate: Pruning ratio for base mask (float, e.g., 0.1 means prune 10%).

    Returns:
        best_acc: Best test accuracy achieved during the process.
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    total_start_time = time.time()
    times_list = []

    # === [Init] Prepare result and log files ===
    initialize_results_file(results_file, base_pruning_rate)
    save_args(args, results_file.replace('results.csv', 'args.txt'))

    # === [Init] Load data and initialize model ===
    trainloader, testloader, n_training, n_test = get_data_loader(args.dataset, batchsize=args.batchsize)
    args.n_training = n_training
    model, init_state, optimizer, loss_fn = initialize_model(args)
    best_acc = 0  # 初始化 best_acc

    # === Stage 1: DP Pretraining & Importance Accumulation ===
    print("==> Stage 1: Pretraining...")
    start_time = time.time()

    pretrain_epochs = 15

    importance_scores = [torch.zeros_like(p, device=device) for p in model.parameters()]
    lr = args.lr

    for epoch in range(pretrain_epochs):
        print(f"[Stage 1] Epoch {epoch + 1}/{pretrain_epochs}")
        model.train()
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
        log_results(results_file, 'Stage 1 (Pretrain)', epoch + 1, train_loss, train_acc, test_loss, test_acc,
                    eps_spent, sigma=noise_multiplier)

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
        model, _ = train(
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
        model, _ = train(
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

