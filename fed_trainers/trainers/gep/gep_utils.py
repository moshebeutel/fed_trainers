import gc
from typing import Optional, Tuple
import numpy as np
import torch
from numpy.core.defchararray import translate
# from sklearn.decomposition import PCA
from torch import Tensor


# def flatten_tensor(tensor_list) -> torch.Tensor:
#
#     # for i in range(len(tensor_list)):
#     #     tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
#     #     # tensor_list[i] = tensor_list[i].reshape(1, -1)
#     flatten_param = torch.stack(tensor_list)
#     flatten_param = flatten_param.reshape(flatten_param.shape[0], -1)
#     return flatten_param
def flatten_tensor(tensor_list) -> torch.Tensor:
    """
    Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
    """
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
        # tensor_list[i] = tensor_list[i].reshape(1, -1)
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


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
# def get_bases(pub_grad, num_bases):
#     num_k = pub_grad.shape[0]
#     num_p = pub_grad.shape[1]
#
#     num_bases = min(num_bases, min(num_p, num_k))
#
#     pca = PCA(n_components=num_bases)
#     pca.fit(pub_grad.cpu().detach().numpy())
#
#     # error_rate = check_approx_error(torch.from_numpy(pca.components_).T, pub_grad)
#
#     return num_bases, pca
#     # return num_bases, error_rate, pca
#
#
# def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int) -> PCA:
#     num_bases: int
#     pub_error: float
#     pca: PCA
#     num_bases, pca = get_bases(basis_gradients, num_basis_elements)
#     # num_bases, pub_error, pca = get_bases(basis_gradients, num_basis_elements)
#     return pca
#
#
# def embed_grad(grad: torch.Tensor, pca: PCA) -> torch.Tensor:
#     grad_np: np.ndarray = grad.cpu().detach().numpy()
#     embedding: np.ndarray = pca.transform(grad_np)
#     return torch.from_numpy(embedding)
#
#
# def project_back_embedding(embedding: torch.Tensor, pca: PCA, device: torch.device) -> torch.Tensor:
#     embedding_np: np.ndarray = embedding.cpu().detach().numpy()
#     grad_np: np.ndarray = pca.inverse_transform(embedding_np)
#     return torch.from_numpy(grad_np).to(device)
#  End of GEP UTILS  numpy variants
#  *************************


#  GEP UTILS  torch variants
#  *************************
@torch.no_grad()
def get_bases(pub_grad, num_bases):
    num_samples = pub_grad.shape[0]
    num_features = pub_grad.shape[1]

    # print(f'num samples: {num_samples} num features: {num_features} num bases: {num_bases}')
    num_bases = min(num_bases, min(num_samples, num_features))

    # print(f'num bases: {num_bases} to compute')

    mean = torch.mean(pub_grad, dim=0, keepdim=True)
    std = torch.std(pub_grad, dim=0, keepdim=True)
    mx, _ = torch.max(pub_grad, dim=0, keepdim=True)
    mn, _ = torch.min(pub_grad, dim=0, keepdim=True)

    # translate_transform = mn
    # translate_transform = float(mn.mean())
    translate_transform = pub_grad.mean(dim=(-2,), keepdim=True)
    # scale_transform = torch.max(torch.tensor(.0001), mx - mn)
    scale_transform = max(0.000001, float(mx.mean()) - float(mn.mean()))
    X = (pub_grad - translate_transform) / scale_transform

    # U, S, V = torch.pca_lowrank(X, q=num_bases, niter=2, center=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # print(f'translate_transform {translate_transform}\nscale_transform {scale_transform}')

    # for nb in [n for n in [10, 20, 50, 100, 500, 750, 1000] if n <= num_bases]:
    # for nb in [n for n in [5, 10, 20, 50, 100, 150, 176]]:
    #     err = torch.dist(X, U[:, :nb] @ torch.diag(S[:nb]) @ Vh[:nb, :])
    #     print(f'Reconstruction Error for num bases {nb}: {err}')

    if torch.any(torch.isnan(Vh)):
        raise Exception(
            f'NaNs in V: {torch.sum(torch.any(torch.isnan(Vh)))} NaNs')

    explained_variance_ = ((S ** 2) / (num_samples - 1)).squeeze()
    total_var = torch.sum(explained_variance_)
    explained_variance_ratio_ = explained_variance_ / total_var

    # for obj in [U,S,Vh,explained_variance_, explained_variance_ratio_]:
    #     obj.detach().cpu()

    explained_variance_ratio_cumsum = torch.cumsum(explained_variance_ratio_, dim=0)
    num_components_explained_variance_ratio_dict = {}
    for th in [0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]:
        over_th = torch.argwhere(explained_variance_ratio_cumsum > th)
        over_th_idx = over_th[0] if len(over_th) > 0 else len(explained_variance_ratio_cumsum) - 1
        num_components_explained_variance_ratio_dict[th] = int(over_th_idx)

    # pca = torch.linalg.qr(pub_grad.t())
    # print(f'Q shape {pca[0].shape}')
    # print(f'R shape {pca[1].shape}')

    S = S.cpu()
    explained_variance_ = explained_variance_.cpu()
    explained_variance_ratio_ = explained_variance_ratio_.cpu()
    explained_variance_ratio_cumsum = explained_variance_ratio_cumsum.cpu()
    # del S, explained_variance_, explained_variance_ratio_, explained_variance_ratio_cumsum

    gc.collect()
    torch.cuda.empty_cache()

    # The principal directions are the transpose of Vh
    V = Vh.t()[:, :num_bases]

    # return V, translate_transform, scale_transform, num_components_explained_variance_ratio_dict
    return V, translate_transform, scale_transform, explained_variance_ratio_, explained_variance_ratio_cumsum


@torch.no_grad()
def embed_grad(grad: torch.Tensor, pca, device=torch.device('cuda')) -> torch.Tensor:
    # embedding: torch.Tensor = torch.matmul(grad, pca)
    # return embedding
    V, translate_transform, scale_transform, _, _ = pca
    # V, grad = V.to(device), grad.to(device)
    with torch.amp.autocast('cuda', enabled=False):
        grad = (grad - translate_transform) / scale_transform
        embedding: torch.Tensor = torch.matmul(grad, V)

    if torch.any(torch.isnan(embedding)):
        raise Exception(
            f'NaNs in embedding: {torch.sum(torch.any(torch.isnan(embedding)))} NaNs')
    # V, grad, embedding = V.detach().cpu(), grad.detach().cpu(), embedding.detach().cpu()
    return embedding


@torch.no_grad()
def project_back_embedding(embedding: torch.Tensor, pca, device) -> torch.Tensor:
    # reconstructed = torch.matmul(embedding, pca.t())
    # return reconstructed
    V, translate_transform, scale_transform, _, _ = pca
    V, embedding = V.to(device), embedding.to(device)
    if torch.any(torch.isnan(embedding)):
        raise Exception(
            f'NaNs in embedding: {torch.sum(torch.any(torch.isnan(embedding)))} NaNs')
    with torch.amp.autocast('cuda', enabled=False):
        reconstructed = torch.matmul(embedding, V.t())
        if torch.any(torch.isnan(reconstructed)):
            raise Exception(
                f'NaNs in reconstructed: {torch.sum(torch.any(torch.isnan(reconstructed)))} NaNs')
        reconstructed = (reconstructed * scale_transform) + translate_transform
    # V, embedding, reconstructed = V.detach().cpu(), embedding.detach().cpu(), reconstructed.detach().cpu()
    return reconstructed


@torch.no_grad()
def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int, device=torch.device('cuda')):
    pca = get_bases(basis_gradients, num_basis_elements)
    return pca


#  End of GEP UTILS  torch variants
#  *************************

@torch.no_grad()
def add_new_gradients_to_history(new_gradients: torch.Tensor,
                                 basis_gradients: Optional[torch.Tensor],
                                 basis_gradients_cpu: Optional[torch.Tensor],
                                 gradients_history_size: int) -> Tuple[Tensor, Tensor, int]:

    basis_gradients_cpu = torch.cat((basis_gradients_cpu, new_gradients), dim=0) \
        if basis_gradients_cpu is not None \
        else new_gradients
    # print(f'\n\t\t\t\t\t\t\t\t2 - basis gradients shape {basis_gradients.shape}')

    basis_gradients_cpu = basis_gradients_cpu[-gradients_history_size:] \
        if gradients_history_size < basis_gradients_cpu.shape[0] \
        else basis_gradients_cpu

    # basis_gradients = basis_gradients_cpu.to('cuda', non_blocking=True)
    basis_gradients = basis_gradients_cpu.to('cuda', non_blocking=True) if new_gradients.device == torch.device(
        'cuda') else basis_gradients_cpu

    filled_history_size = basis_gradients_cpu.shape[0]

    return basis_gradients, basis_gradients_cpu, filled_history_size
