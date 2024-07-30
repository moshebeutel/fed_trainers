import argparse
import copy
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import wandb
from sklearn import metrics
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm import trange
from model import FeatureModel
from utils import set_logger, set_seed, str2bool, initialize_weights, get_device


def get_clients(args):
    num_clients = args.num_clients
    num_private_clients = args.num_private_clients
    num_public_clients = args.num_public_clients

    assert num_clients >= (num_private_clients + num_public_clients), \
        f'num clients should be more than sum of all participating clients. Got {num_clients} clients'

    num_dummy_clients = num_clients - (num_private_clients + num_public_clients)

    i = 0
    public_clients = list(range(i, i + num_public_clients))
    i += num_public_clients
    private_clients = list(range(i, i + num_private_clients))
    i += num_private_clients
    dummy_clients = list(range(i, i + num_dummy_clients))
    i += num_dummy_clients

    return public_clients, private_clients, dummy_clients


def get_optimizer(args, network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


def get_dataloaders(args):
    import pandas as pd
    from biolab_utilities.putemg_utilities import prepare_data, Record, record_filter, data_per_id_and_date


    # filtered_data_folder = os.path.join(result_folder, 'filtered_data')
    # calculated_features_folder = os.path.join(result_folder, 'calculated_features')
    calculated_features_folder = Path('/home/user/GIT/putemg-downloader/Data-HDF5-Features-Small')

    # list all hdf5 files in given input folder
    all_files = [f.as_posix().replace('_filtered_features', '') for f in sorted(calculated_features_folder.glob("*_features.hdf5"))]

    all_feature_records = [Record(os.path.basename(f)) for f in all_files]

    records_filtered_by_subject = record_filter(all_feature_records)

    splits_all = data_per_id_and_date(records_filtered_by_subject, n_splits=3)

    # data can be additionally filtered based on subject id

    # records_filtered_by_subject = record_filter(all_feature_records,
    #                                             whitelists={"id": ["01", "02", "03", "04", "07"]})
    # records_filtered_by_subject = pu.record_filter(all_feature_records, whitelists={"id": ["01"]})

    # load feature data to memory
    dfs: Dict[Record, pd.DataFrame] = {}

    for r in records_filtered_by_subject:
        # print("Reading features for input file: ", r)
        filename = os.path.splitext(r.path)[0]
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder,
                                                       filename + '_filtered_features.hdf5')))



    features = ['RMS', 'MAV', 'WL', 'ZC', 'SSC', 'IAV', 'VAR', 'WAMP']
    # defines gestures to be used in shallow learn
    gestures = {
        0: "Idle",
        1: "Fist",
        2: "Flexion",
        3: "Extension",
        4: "Pinch index",
        5: "Pinch middle",
        6: "Pinch ring",
        7: "Pinch small"
    }
    channel_range = {
        "24chn": {"begin": 1, "end": 24},
        # "8chn_1band": {"begin": 1, "end": 8},
        "8chn_2band": {"begin": 9, "end": 16},
        # "8chn_3band": {"begin": 17, "end": 24}
    }
    ch_range = channel_range['24chn']
    num_classes = 8
    classes_per_client = 8
    num_clients = len(splits_all.values())
    train_loaders, test_loaders = {}, {}
    for client_id in range(num_clients):
        running_loss, running_correct, running_samples = 0., 0., 0.

        # iterate over each internal data
        for i_s, subject_data in enumerate(list(splits_all.values())[client_id]):
            is_first_iter = True
            # get data of client
            # prepare training and testing set based on combination of k-fold split, feature set and gesture set
            # this is also where gesture transitions are deleted from training and test set
            # only active part of gesture performance remains
            data = prepare_data(dfs, subject_data, features, list(gestures.keys()))

            # list columns containing only feature data
            regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
            cols = list(filter(regex.search, list(data["train"].columns.values)))

            # strip columns to include only selected channels, eg. only one band
            cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]

            # extract limited training x and y, only with chosen channel configuration
            train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
            train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())
            train_y[train_y > 5] -= 2

            # # extract limited testing x and y, only with chosen channel configuration
            test_x = torch.tensor(data["test"][cols].to_numpy(), dtype=torch.float32)
            test_y_true = torch.LongTensor(data["test"]["output_0"].to_numpy())
            test_y_true[test_y_true > 5] -= 2

            train_loaders[client_id] = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x, train_y),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            test_loaders[client_id] = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_x, test_y_true),
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

    return train_loaders, test_loaders, test_loaders


def get_model(args):
    num_classes = {'cifar10': 10, 'cifar100': 100, 'putEMG': 8}[args.data_name]
    assert args.data_name == 'putEMG', 'data_name should be putEMG'
    assert num_classes == 8, 'num_classes should be 8'
    model = FeatureModel(num_channels=24, num_features=8, number_of_classes=num_classes)
    initialize_weights(model)
    return model


@torch.no_grad()
def get_dp_noise(args) -> torch.Tensor:
    noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                         size=(args.num_steps, args.num_client_agg, args.basis_gradients_history_size))
    return noise


# GEP

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


@torch.no_grad()
def eval_model(args, global_model, client_ids, loaders):
    device = get_device()
    # device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    loss_dict: Dict[str, float] = {}
    acc_dict: Dict[str, float] = {}
    acc_score_dict: Dict[str, float] = {}
    f1s_dict: Dict[str, float] = {}
    criteria = torch.nn.CrossEntropyLoss()

    y_true_all, y_pred_all, loss_all = None, None, 0.

    global_model.eval()
    num_clients = len(client_ids)

    for i, client_id in enumerate(client_ids):
        running_loss, running_correct, running_samples = 0., 0., 0.

        test_loader = loaders[client_id]

        all_targets = []
        all_preds = []

        for batch_count, batch in enumerate(test_loader):
            X_test, Y_test = tuple(t.to(device) for t in batch)
            pred = global_model(X_test)

            loss = criteria(pred, Y_test)
            predicted = torch.max(pred, dim=1)[1].cpu().numpy()

            running_loss += (loss.item() * Y_test.size(0))
            running_correct += pred.argmax(1).eq(Y_test).sum().item()
            running_samples += Y_test.size(0)

            target = Y_test.cpu().numpy().reshape(predicted.shape)

            all_targets += target.tolist()
            all_preds += predicted.tolist()

        # calculate confusion matrix
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        running_loss /= running_samples

        y_true_all = y_true if y_true_all is None else np.concatenate((y_true_all, y_true), axis=0)
        y_pred_all = y_pred if y_pred_all is None else np.concatenate((y_pred_all, y_pred), axis=0)
        loss_all += (running_loss / num_clients)

        eval_accuracy = (y_true == y_pred).sum().item() / running_samples
        acc_score = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='micro')

        acc_dict[f"P{client_id}"] = eval_accuracy
        loss_dict[f"P{client_id}"] = running_loss
        acc_score_dict[f"P{client_id}"] = acc_score
        f1s_dict[f"P{client_id}"] = f1

    avg_acc = (y_true_all == y_pred_all).mean().item()
    avg_loss = loss_all
    avg_acc_score = metrics.accuracy_score(y_true_all, y_pred_all)
    avg_f1 = metrics.f1_score(y_true_all, y_pred_all, average='micro')

    return acc_dict, loss_dict, acc_score_dict, f1s_dict, avg_acc, avg_loss, avg_acc_score, avg_f1


def train(args):
    fields_list = ["num_blocks", "block_size", "optimizer", "lr",
                   "num_client_agg", "clip", "noise_multiplier", "basis_gradients_history_size"]

    args_list = [(k, vars(args)[k]) for k in fields_list]

    logging.info(f' *** Training for args {args_list} ***')

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1 = 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    public_clients, private_clients, dummy_clients = get_clients(args)
    device = get_device()
    # device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)
    criteria = torch.nn.CrossEntropyLoss()

    dp_noise: torch.Tensor = get_dp_noise(args).to(device)

    basis_gradients: Optional[torch.Tensor] = None

    train_loaders, val_loaders, test_loaders = get_dataloaders(args)

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    step_iter = trange(args.num_steps)

    for step in step_iter:

        # initialize global model params
        params = OrderedDict()
        grads = OrderedDict()
        # public_params = OrderedDict()
        public_grads = OrderedDict()
        prev_params = OrderedDict()
        for n, p in net.named_parameters():
            params[n] = torch.zeros_like(p.data)
            grads[n] = []
            # public_params[n] = torch.zeros_like(p.data)
            public_grads[n] = []
            prev_params[n] = p.detach()

        # iterate over each client
        train_avg_loss = 0

        # local trains on public clients - get gradients for subspace
        for j, c_id in enumerate(public_clients):

            train_loader = train_loaders[c_id]

            description_pref = f"Step: {step + 1:2d}, Public client: {c_id:2d} ({j + 1} in public clients),"

            local_net, train_avg_loss = local_train(args, net, train_loader, train_avg_loss, step_iter,
                                                    description_pref)

            # get client grads and sum.
            for n, p in local_net.named_parameters():
                # public_params[n] += p.data
                public_grads[n].append(p.data.detach() - prev_params[n])

        public_grads_list = [torch.stack(public_grads[n]) for n, p in net.named_parameters()]

        public_grads_flattened = flatten_tensor(public_grads_list)
        pca = update_subspace(args, basis_gradients, public_grads_flattened)

        train_avg_loss = 0
        # select several clients
        # client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)
        client_ids_step = np.random.choice([*public_clients, *private_clients], size=args.num_client_agg, replace=False)
        # local trains on sampled clients
        for j, c_id in enumerate(client_ids_step):

            train_loader = train_loaders[c_id]

            description_pref = f"Step: {step + 1:2d}, client: {c_id:2d} ({j + 1} in step),"
            description_suff = (f"best epoch: {best_epoch + 1: 2d},"
                                f" test avg acc: {val_avg_acc:.4f}, "
                                f"best test acc: {best_acc:.2f}")

            local_net, train_avg_loss = local_train(args, net, train_loader, train_avg_loss, step_iter,
                                                    description_pref, description_suff)

            # get client grads and sum.
            for n, p in local_net.named_parameters():
                params[n] += p.data
                grads[n].append(p.data.detach() - prev_params[n])

        grads_list = [torch.stack(grads[n]) for n, p in net.named_parameters()]

        grads_flattened = flatten_tensor(grads_list)
        embedded_grads = embed_grad(grads_flattened, pca).to(device)
        noised_embedded_grads = embedded_grads + dp_noise[step, :, :embedded_grads.shape[-1]]
        aggregated_noised_embedded_grads = torch.sum(noised_embedded_grads, dim=0)
        reconstructed_grad = project_back_embedding(aggregated_noised_embedded_grads, pca, device)
        # average parameters
        offset = 0
        for n, p in params.items():
            params[n] = (p + reconstructed_grad[offset: offset + p.numel()].reshape(p.shape)) / args.num_client_agg
            offset += p.numel()

        # update new parameters of global net
        net.load_state_dict(params)

        if step % args.eval_every == 0 or (step + 1) == args.num_steps:
            val_results = eval_model(args, net, private_clients, val_loaders)

            val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict, \
                val_avg_acc, val_avg_loss, val_avg_acc_score, val_avg_f1 = val_results

            if val_avg_acc > best_acc:
                best_acc = val_avg_acc
                best_loss = val_avg_loss
                best_acc_score = val_avg_acc_score
                best_f1 = val_avg_f1
                best_epoch = step
                best_model = best_model.cpu()
                del best_model
                best_model = copy.deepcopy(net)

        if args.wandb:
            log_dict = {}
            log_dict.update(
                {
                    'custom_step': step,
                    'train_loss': train_avg_loss,
                    'test_avg_loss': val_avg_loss,
                    'test_avg_acc': val_avg_acc,
                    'test_avg_acc_score': val_avg_acc_score,
                    'test_avg_f1': val_avg_f1,
                    'test_best_loss': best_loss,
                    'test_best_acc': best_acc,
                    'test_best_acc_score': best_acc_score,
                    'test_best_f1': best_f1,
                    'test_best_epoch': best_epoch
                }
            )
            log_dict.update({f"test_acc_{l}": m for (l, m) in val_acc_dict.items()})
            log_dict.update({f"test_loss_{l}": m for (l, m) in val_loss_dict.items()})
            log_dict.update({f"test_acc_score_{l}": m for (l, m) in val_acc_score_dict.items()})
            log_dict.update({f"test_f1_{l}": m for (l, m) in val_f1s_dict.items()})

            wandb.log(log_dict)

        logging.debug(
            f"epoch {step}, "
            f"train loss {train_avg_loss:.3f}, "
            f"best epoch: {best_epoch}, "
            f"best test loss: {best_loss:.3f}, "
            f"best test acc: {best_acc:.3f}, "
            f"test test acc score: {best_acc_score:.3f}, "
            f"best test f1: {best_f1:.3f}"
        )

    # Test best model
    test_results = eval_model(args, best_model, private_clients, test_loaders)

    _, _, _, _, test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = test_results

    logging.info(f'### Test Results For Args {args_list}:'
                 f' test acc {test_avg_acc:.4f},'
                 f' test loss {test_avg_loss:.4f} ###')


def update_subspace(args, basis_gradients, grads_flattened):
    basis_gradients = add_new_gradients_to_history(grads_flattened, basis_gradients,
                                                   args.basis_gradients_history_size)
    pca = compute_subspace(basis_gradients, args.basis_gradients_history_size)
    return pca


def local_train(args, net, train_loader, train_avg_loss, step_iter, desc_pref='', desc_suff=''):
    local_net = copy.deepcopy(net)
    local_net.train()
    optimizer = get_optimizer(args, local_net)
    criteria = torch.nn.CrossEntropyLoss()
    device = get_device()
    for i in range(args.inner_steps):
        for k, batch in enumerate(train_loader):
            # batch = next(iter(train_loader))
            x, Y = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # forward prop
            pred = local_net(x)
            loss = criteria(pred, Y)

            # back prop
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), args.clip)
            # update local parameters
            optimizer.step()

            # aggregate losses
            train_avg_loss += (loss.item() / Y.shape[0])

            step_iter.set_description(
                f"{desc_pref}"
                f"Inner Step: {i + 1:2d},"
                f"batch: {k + 1:2d},"
                f"Loss: {loss.item() : 5.2f},"
                f"train loss {train_avg_loss: 5.2f},"
                f"{desc_suff}"
            )
        # end of for k, batch in enumerate(train_loader):
    # end of for i in range(args.inner_steps):
    return local_net, train_avg_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Personalized Federated Learning")

    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default='adam',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num-client-agg", type=int, default=5, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=0.1, help="gradient clip")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="dp noise factor "
                                                                            "to be multiplied by clip")

    ##################################
    #       GEP args                 #
    ##################################
    parser.add_argument("--basis-gradients-history-size", type=int,
                        default=100, help="amount of past gradients participating in embedding subspace computation")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=False)

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="putEMG",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dataset name"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=23, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=18, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=5, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=2, help="number of simulated clients")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected epochs")

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    exp_name = f'FedAvg_between-days_seed_{args.seed}_wd_{args.wd}_' \
               f'lr_{args.lr}_num-steps_{args.num_steps}_inner-steps_{args.inner_steps}'

    # Weights & Biases
    if args.wandb:
        wandb.init(project="key_press_emg_toronto", name=exp_name)
        wandb.config.update(args)

    train(args)
