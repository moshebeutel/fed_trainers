from typing import Optional, Tuple
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


#  GEP UTILS  numpy variants
#  *************************
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
#  End of GEP UTILS  numpy variants
#  *************************


#  GEP UTILS  torch variants
#  *************************
# @torch.no_grad()
# def get_bases(pub_grad, num_bases) -> Tuple[int, torch.Tensor]:
#     num_k = pub_grad.shape[0]
#     num_p = pub_grad.shape[1]
#
#     num_bases = min(num_bases, min(num_p, num_k))
#
#     pca = torch.pca_lowrank(pub_grad, q=num_bases, niter=10)
#     # error_rate = check_approx_error(pca[-1], pub_grad)
#
#     # print(f'\n\t\t\t\t\t\t\t\tnum_bases {num_bases}\tPCA error: {error_rate}')
#
#     return num_bases, pca[-1]
#
#
# def embed_grad(grad: torch.Tensor, pca: torch.Tensor) -> torch.Tensor:
#     embedding: torch.Tensor = torch.matmul(grad, pca)
#     return embedding
#
#
# def project_back_embedding(embedding: torch.Tensor, pca: torch.Tensor, device) -> torch.Tensor:
#     reconstructed = torch.matmul(embedding, pca.t())
#     return reconstructed
#
#
# @torch.no_grad()
# def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int) -> torch.Tensor:
#     pca: torch.Tensor
#     _, pca = get_bases(basis_gradients, num_basis_elements)
#     return pca

#  End of GEP UTILS  torch variants
#  *************************

@torch.no_grad()
def add_new_gradients_to_history(new_gradients: torch.Tensor, basis_gradients: Optional[torch.Tensor],
                                 gradients_history_size: int) -> Tensor:
    # print(f'\n\t\t\t\t\t\t\t\t1 - basis gradients shape {basis_gradients.shape if basis_gradients is not None else None}')

    basis_gradients = torch.cat((basis_gradients, new_gradients), dim=0) \
        if basis_gradients is not None \
        else new_gradients
    # print(f'\n\t\t\t\t\t\t\t\t2 - basis gradients shape {basis_gradients.shape}')

    basis_gradients = basis_gradients[-gradients_history_size:] \
        if gradients_history_size < basis_gradients.shape[0] \
        else basis_gradients

        # print(f'\n\t\t\t\t\t\t\t\t3 - basis gradients shape {basis_gradients.shape}')

    return basis_gradients

