from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor


@torch.no_grad()
def check_approx_error(L, target) -> float:
    L = L.to(target.device)
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))

    return -1.0 if target.item() == 0 else error.item() / target.item()


def get_bases(pub_grad, num_bases):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, min(num_p, num_k))

    pca = PCA(n_components=num_bases)
    pca.fit(pub_grad.cpu().detach().numpy())

    error_rate = check_approx_error(torch.from_numpy(pca.components_).T, pub_grad)

    return num_bases, error_rate, pca


def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int) -> PCA:
    num_bases: int
    pub_error: float
    pca: PCA
    num_bases, pub_error, pca = get_bases(basis_gradients, num_basis_elements)
    return pca


def embed_grad(grad: torch.Tensor, pca: PCA) -> torch.Tensor:
    grad_np: np.ndarray = grad.cpu().detach().numpy()
    embedding: np.ndarray = pca.transform(grad_np)
    return torch.from_numpy(embedding)


def project_back_embedding(embedding: torch.Tensor, pca: PCA, device: torch.device) -> torch.Tensor:
    embedding_np: np.ndarray = embedding.cpu().detach().numpy()
    grad_np: np.ndarray = pca.inverse_transform(embedding_np)
    return torch.from_numpy(grad_np).to(device)


def add_new_gradients_to_history(new_gradients: torch.Tensor, basis_gradients: Optional[torch.Tensor],
                                 basis_gradients_history_size: int) -> Tensor:
    basis_gradients = torch.cat((basis_gradients, new_gradients), dim=0) \
        if basis_gradients is not None \
        else new_gradients
    basis_gradients = basis_gradients[-basis_gradients_history_size:] \
        if basis_gradients_history_size < basis_gradients.shape[0] \
        else basis_gradients
    return basis_gradients


def update_subspace(args, basis_gradients, grads_flattened):
    basis_gradients = add_new_gradients_to_history(grads_flattened, basis_gradients,
                                                   args.basis_gradients_history_size)
    pca = compute_subspace(basis_gradients, int(args.basis_gradients_history_size * 0.8))
    return pca
