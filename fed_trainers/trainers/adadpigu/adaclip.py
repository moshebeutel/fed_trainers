import torch
from backpack import backpack
from backpack.extensions import BatchGrad
def SparseAdaCliP(model, batch_x, batch_y, loss_fn,
                  m_vec, s_vec, clip_norm, noise_multiplier,
                  topk_ratio=1.0, mask=None):
    """
    Sparse AdaCliP: Coordinate-wise adaptive clipping + Top-k sparsification + structural mask.
    Args:
        model: PyTorch model (parameters extended by BackPACK).
        batch_x, batch_y: Input batch and labels.
        loss_fn: Loss function (should be extended by BackPACK).
        m_vec, s_vec: Running mean and variance vectors for each coordinate (for AdaCliP normalization).
        clip_norm: Clipping threshold (L2 norm).
        noise_multiplier: DP Gaussian noise multiplier.
        topk_ratio: Fraction of coordinates to update each round (0 < r <= 1).
        mask: List of binary mask tensors, same shape as model parameters, controls structure (optional).
    Returns:
        final_grad: Flattened noisy gradient for parameter update.
        recovered_grad: Per-sample recovered gradient (for m_vec, s_vec update).
    """
    device = batch_x.device
    model.train()

    outputs = model(batch_x)
    loss = loss_fn(outputs, batch_y)

    with backpack(BatchGrad()):
        loss.backward()

    B = batch_x.size(0)
    all_grads = []
    for param in model.parameters():
        if hasattr(param, 'grad_batch'):
            grad_batch = param.grad_batch.reshape(B, -1)
            all_grads.append(grad_batch)
    grad_matrix = torch.cat(all_grads, dim=1)  # [B, D]

    # === [1] Apply structure mask (keep only allowed coordinates) ===
    if mask is not None:
        flat_mask = torch.cat([m.flatten() for m in mask]).to(device)  # [D]
        grad_matrix = grad_matrix * flat_mask.unsqueeze(0)  # mask.shape: [1, D]

    # === [2] Coordinate-wise normalization (zero mean, unit variance) ===
    centered_grad = (grad_matrix - m_vec) / (s_vec.sqrt() + 1e-6)  # [B, D]

    # === [3] Top-k sparsification (only within structure mask) ===
    if topk_ratio < 1.0:
        total_dim = centered_grad.shape[1]
        k = int(total_dim * topk_ratio)
        abs_centered = centered_grad.abs()
        threshold, _ = torch.kthvalue(abs_centered, total_dim - k + 1, dim=1, keepdim=True)
        sparse_mask = (abs_centered >= threshold)
        centered_grad = centered_grad * sparse_mask  # [B, D]

    # === [4] L2 norm clipping (across each sample) ===
    norms = torch.norm(centered_grad, dim=1)  # [B]
    clip_factors = (clip_norm / (norms + 1e-6)).clamp(max=1.0)  # [B]
    centered_grad = centered_grad * clip_factors.unsqueeze(1)

    # === [5] Add Gaussian noise (to the mean) for DP ===
    clipped_mean = centered_grad.mean(dim=0)
    sigma = noise_multiplier*clip_norm/B
    noisy_mean = clipped_mean + torch.randn_like(clipped_mean)*sigma
    #noise = torch.randn_like(centered_grad) * (noise_multiplier * clip_norm)
    #noisy_grad = centered_grad + noise / B

    # === [6] Restore original scale (inverse AdaCliP transform) ===
    final_grad = noisy_mean * (s_vec.sqrt()+1e-6) + m_vec
    #recovered_grad = noisy_grad * (s_vec.sqrt() + 1e-6) + m_vec  # [B, D]

    # === [7] Per-sample recovered gradient (for running statistics update) ===
    recovered_grad = centered_grad *  (s_vec.sqrt() + 1e-6) + m_vec
    #final_grad = recovered_grad.mean(dim=0)  # [D]

    # === [8] Defensive: zero out coordinates outside mask again ===
    if mask is not None:
        final_grad = final_grad * flat_mask  # [D]

    return final_grad, recovered_grad

def update_m_s(recovered_mean, m_vec, s_vec, noise_multiplier, beta_1=0.9, beta_2=0.999, h_1=1e-6, h_2=1e6):
    """
    Update running mean (m_vec) and variance (s_vec) for coordinate-wise adaptive clipping (AdaCliP).
    Uses exponential moving average for both mean and variance.

    Args:
        recovered_mean: Tensor of per-coordinate batch mean gradient (after de-normalization), shape [D].
        m_vec: Previous running mean vector, shape [D].
        s_vec: Previous running variance vector, shape [D].
        noise_multiplier: Gaussian noise multiplier for DP (unused by default, can be used for variance correction).
        beta_1: Decay rate for mean moving average (default: 0.9).
        beta_2: Decay rate for variance moving average (default: 0.999).
        h_1: Minimum allowed variance value (default: 1e-6).
        h_2: Maximum allowed variance value (default: 1e6).

    Returns:
        m_vec: Updated mean vector, shape [D].
        s_vec: Updated variance vector, shape [D].
    """
    # === Save the old mean vector (for variance estimation) ===
    m_vec_old = m_vec.clone()

    # === Update running mean (EMA) ===
    m_vec = beta_1 * m_vec + (1 - beta_1) * recovered_mean

    # === Estimate variance as squared difference from previous mean ===
    grad_diff = (recovered_mean - m_vec_old) ** 2
    var_est = grad_diff

    # === Clamp the variance to avoid numerical issues ===
    var_est = torch.clamp(var_est, min=h_1, max=h_2)

    # === Update running variance (EMA) ===
    s_vec = beta_2 * s_vec + (1 - beta_2) * var_est

    return m_vec, s_vec