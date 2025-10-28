import copy
import gc
from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np
import torch
from numpy.core.defchararray import translate
# from sklearn.decomposition import PCA
from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import initialize_weights
from utils import get_device, get_optimizer


class Unet1dAutoencoder(nn.Module):
    def __init__(self, input_length, num_layers=6, latent_dim=44, base_channels=64):
        """
        Args:
            input_length (int): Length of the input 1D tensor.
            num_layers (int): Number of encoder/decoder layers.
            latent_dim (int): Size of the latent representation.
            base_channels (int): Number of channels in the first conv layer.
        """
        super(Unet1dAutoencoder, self).__init__()

        self.input_length = input_length
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # # Calculate the size of each downsampling step
        # self.downsample_sizes = [input_length]
        # for _ in range(num_layers):
        #     self.downsample_sizes.append(self.downsample_sizes[-1] // 2)

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = 1
        out_channels = base_channels

        for _ in range(num_layers):
            self.encoder.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0))
            in_channels = out_channels
            out_channels *= 2

            # Compute the output size after encoding
            dummy_input = torch.zeros(1, 1, input_length)
            with torch.no_grad():
                x = dummy_input
                for layer in self.encoder:
                    x = torch.relu(layer(x))
                encoded = x
                self.encoded_channels = encoded.shape[1]
                self.encoded_length = encoded.shape[2]
                encoded_size = self.encoded_channels * self.encoded_length

            dummy_input = dummy_input.cpu().detach()
            x = x.cpu().detach()
            dummy_input = None
            x = None
            del dummy_input, x
            gc.collect()
            torch.cuda.empty_cache()

        # Bottleneck (Latent Space)
        self.bottleneck_enc = nn.Linear(encoded_size, latent_dim)
        self.bottleneck_dec = nn.Linear(latent_dim, encoded_size)

        # Decoder
        self.decoder = nn.ModuleList()
        in_channels = self.encoded_channels
        out_channels = in_channels // 2

        for _ in range(num_layers):
            self.decoder.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0, output_padding=1))
            in_channels = out_channels
            out_channels //= 2

        # Final output layer
        self.output_layer = nn.ConvTranspose1d(in_channels, 1, kernel_size=3, stride=3, padding=0, output_padding=1)

        initialize_weights(self)


    def encode(self, x):
        # print('encode')
        # print(x.shape)
        batch_size = x.size(0)
        # Encoding
        for i,layer in enumerate(self.encoder):
            x = torch.relu(layer(x))
            # print(f'layer {i} shape {x.shape}')

        # Latent space
        x_flat = x.view(batch_size, -1)
        # print(f'x_flat shape {x_flat.shape}')
        latent = self.bottleneck_enc(x_flat)
        # print(f'latent shape {latent.shape}')
        return latent

    def decode(self, latent):
        batch_size = latent.size(0)
        # print('decode')
        # print(latent.shape)
        # unflat
        latent = self.bottleneck_dec(latent)
        # print(f'latent after bottleneck dec {latent.shape}')
        x = latent.view(batch_size, self.encoded_channels, self.encoded_length)
        # print(f'latent after unflat {x.shape}')
        # Decoding
        for i, layer in enumerate(self.decoder):
            x = torch.relu(layer(x))
            # print(f'layer {i} shape {x.shape}')

        # Output reconstruction
        output = self.output_layer(x)
        # print(output.shape)
        # Ensure the output matches the input size exactly
        output = output[..., :self.input_length]
        # print(output.shape)
        return output


    def forward(self, x):

        latent = self.encode(x)

        # Output reconstruction
        output = self.decode(latent)

        return output, latent




def train_nonlinear_dim_reduction(args, public_loaders: list[DataLoader], net: nn.Module) -> None:

    device = get_device()
    # initialize global model params
    public_grads = OrderedDict()
    prev_params = OrderedDict()
    for n, p in net.named_parameters():
        public_grads[n] = []
        prev_params[n] = p.detach().to(device)

    multiloader = DataLoader(
            torch.utils.data.ConcatDataset([loader.dataset for loader in public_loaders]),
            batch_size=public_loaders[0].batch_size,
            shuffle=True
        )

    criteria = torch.nn.CrossEntropyLoss()
    # for loader in public_loaders:
    local_net: torch.nn.Module = copy.deepcopy(net)
    local_net.to(device)
    local_net.train()
    optimizer = torch.optim.SGD(local_net.parameters(), lr=1.0, weight_decay=0.1, momentum=0.9)
    for batch in multiloader:
        x, Y = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        pred = local_net(x)
        loss = criteria(pred, Y)
        loss.backward()

        optimizer.step()

        # get client grads and sum.
        for n, p in local_net.named_parameters():
            public_grads[n].append(p.data.detach() - prev_params[n])
            prev_params[n] = p.detach()

        x, Y, pred, loss = x.detach().cpu(), Y.detach().cpu(), pred.detach().cpu(), loss.detach().cpu()
        x, Y, pred, loss = None, None, None, None
        del x, Y, pred, loss
        gc.collect()
        torch.cuda.empty_cache()

    local_net = local_net.to('cpu')
    local_net = None
    del local_net
    gc.collect()
    torch.cuda.empty_cache()

    public_grads_list = [torch.stack(public_grads[n]) for n, p in net.named_parameters()]

    public_grads_flat = flatten_tensor(public_grads_list)
    # Create a TensorDataset and DataLoader from public_grads_flat rows
    dataset = torch.utils.data.TensorDataset(public_grads_flat)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


    public_grads_list =[t.detach().cpu() for t in public_grads_list]
    public_grads_flat = public_grads_flat.detach().cpu()
    dataset = torch.utils.data.TensorDataset(public_grads_flat)
    public_grads_flat, public_grads_list, dataset = None, None, None
    del public_grads_flat, public_grads_list, dataset
    gc.collect()
    torch.cuda.empty_cache()



    autoencoder = get_autoencoder()
    autoencoder = autoencoder.to(device)
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    # autoencoder_loss_fn = nn.CosineEmbeddingLoss()
    autoencoder_loss_fn = nn.MSELoss()
    autoencoder.train()
    for epoch in range(5):
        for i,batch in enumerate(dataloader):
            data = batch[0].to(device)

            autoencoder_optimizer.zero_grad()

            reconstructed, latent = autoencoder(data.unsqueeze(1))

            # Trim reconstructed tensor to match data size
            reconstructed = reconstructed[..., :data.size(-1)]

            autoencoder_loss = autoencoder_loss_fn(reconstructed.squeeze(), data)  # Minimize reconstruction error
            # autoencoder_loss = autoencoder_loss_fn(reconstructed.squeeze(), data, torch.ones(reconstructed.shape[0]).to(device))  # Minimize reconstruction error


            data, reconstructed, latent = data.detach().cpu(), reconstructed.detach().cpu(), latent.detach().cpu()
            data, reconstructed, latent = None, None, None
            del data, reconstructed, latent
            gc.collect()
            torch.cuda.empty_cache()


            autoencoder_loss.backward()
            autoencoder_optimizer.step()
            # print(f'Epoch {epoch} iter {i} autoencoder_loss {autoencoder_loss.item()}')
    get_autoencoder.autoencoder = autoencoder.to('cpu')



def get_autoencoder():
    if not hasattr(get_autoencoder, 'autoencoder'):
        get_autoencoder.autoencoder = Unet1dAutoencoder(input_length=123548)

    return get_autoencoder.autoencoder

@torch.no_grad()
def nonlinear_embed_grad(grad: torch.Tensor) -> torch.Tensor:
    autoencoder = get_autoencoder()
    autoencoder.to(grad.device)
    autoencoder.eval()
    with torch.no_grad():
        embedding = autoencoder.encode(grad.unsqueeze(1))
        return embedding
@torch.no_grad()
def nonlinear_project_back_embedding(embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    autoencoder = get_autoencoder()
    autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        grad = autoencoder.decode(embedding)
        return grad



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


if __name__ == '__main__':
    input_length = 123548
    # layers = [4, 6, 8]
    layers_list = [4, 6]
    for l in layers_list:
        ae = Unet1dAutoencoder(input_length=input_length, num_layers=l, latent_dim=44)
        param_list = [p.numel() for p in ae.parameters()]
        print(f'num layers: {l} num params: {sum(param_list)}, params: {param_list}')
        dummy_input = torch.randn(1, 1, input_length)
        output1, output2 = ae(dummy_input)
        print(output1.shape, output2.shape)
        print(output2)



