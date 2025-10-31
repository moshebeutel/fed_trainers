import torch
import os
import json
import csv
from backpack import backpack
from backpack.extensions import BatchGrad




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