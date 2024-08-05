import copy
import random
import logging
import argparse
import time
from contextlib import contextmanager
import os
import json
from pathlib import Path
import sys
import warnings
from typing import Dict

import numpy as np
import torch
from sklearn import metrics
from torch import nn


def get_n_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number_params = sum([np.prod(p.size()) for p in model_parameters])
    return number_params


def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()


def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(args):
    logger = logging.getLogger(args.log_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{args.log_name}_{time.asctime()}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_device(cuda=True, gpus='0'):
    # return torch.device("cuda:" + gpus if torch.cuda.is_available() and cuda else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def take(X, Y, classes):
    indices = np.isin(Y, classes)
    return X[indices], Y[indices]


def pytorch_take(X, Y, classes):
    indices = torch.stack([y_ == Y for y_ in classes]).sum(0).bool()
    return X[indices], Y[indices]


def lbls1_to_lbls2(Y, l2l):
    for (lbls1_class, lbls2_class) in l2l.items():
        if isinstance(lbls2_class, list):
            for c in lbls2_class:
                Y[Y == lbls1_class] = c + 1000
        elif isinstance(lbls2_class, int):
            Y[Y == lbls1_class] = lbls2_class + 1000
        else:
            raise NotImplementedError("not a valid type")

    return Y - 1000


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# create folders for saving models and logs
def _init_(out_path, exp_name):
    script_path = os.path.dirname(__file__)
    script_path = '.' if script_path == '' else script_path
    if not os.path.exists(out_path + '/' + exp_name):
        os.makedirs(out_path + '/' + exp_name)
    # save configurations
    os.system('cp -r ' + script_path + '/*.py ' + out_path + '/' + exp_name)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [
        int(x.as_posix().split('_')[-1])
        for x in art_dir.iterdir() if x.is_dir()
    ]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(
        vars(args),
        open(out_dir / "meta.experiment", "w")
    )

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def topk(true, pred, k):
    max_pred = np.argsort(pred, axis=1)[:, -k:]  # take top k
    two_d_true = np.expand_dims(true, 1)  # 1d -> 2d
    two_d_true = np.repeat(two_d_true, k, axis=1)  # repeat along second axis
    return (two_d_true == max_pred).sum() / true.shape[0]


def to_one_hot(y, dtype=torch.double):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], y.max().type(torch.IntTensor) + 1), dtype=dtype, device=y.device)
    return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)


def CE_loss(y, y_hat, num_classes, reduction='mean'):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], num_classes), dtype=y_hat.dtype, device=y.device)
    y_output_onehot.scatter_(1, y.unsqueeze(1), 1)
    if reduction == 'mean':
        return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12), dim=1).mean()
    return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12))


def permute_data_lbls(data, labels):
    perm = np.random.permutation(data.shape[0])
    return data[perm], labels[perm]


def N_vec(y):
    """
    Compute the count vector for PG Multinomial inference
    :param x:
    :return:
    """
    if y.dim() == 1:
        N = torch.sum(y)
        reminder = N - torch.cumsum(y)[:-2]
        return torch.cat((torch.tensor([N]).to(y.device), reminder))
    elif y.dim() == 2:
        N = torch.sum(y, dim=1, keepdim=True)
        reminder = N - torch.cumsum(y, dim=1)[:, :-2]
        return torch.cat((N, reminder), dim=1)
    else:
        raise ValueError("x must be 1 or 2D")


def kappa_vec(y):
    """
    Compute the kappa vector for PG Multinomial inference
    :param x:
    :return:
    """
    if y.dim() == 1:
        return y[:-1] - N_vec(y) / 2.0
    elif y.dim() == 2:
        return y[:, :-1] - N_vec(y) / 2.0
    else:
        raise ValueError("x must be 1 or 2D")


# modified from:
# https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


def print_calibration(ECE_module, out_dir, lbls_vs_target, file_name, color, temp=1.0):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(),
                                     color=color, temp=temp)
    logging.info(f"{file_name}, "
                 f"ECE: {ece_metrics[0].item():.3f}, "
                 f"MCE: {ece_metrics[1].item():.3f}, "
                 f"BRI: {ece_metrics[2].item():.3f}")


def calibration_search(ECE_module, out_dir, lbls_vs_target, color, file_name):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    temps = torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    eces = [ECE_module.forward(probs, targets, None, color=color, temp=t)[0].item() for t in temps]
    best_temp = round(temps[np.argmin(eces)].item(), 2)

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(),
                                     color=color, temp=best_temp)
    logging.info(f"{file_name}, "
                 f"Best Temperature: {best_temp:.3f}, "
                 f"ECE: {ece_metrics[0].item():.3f}, "
                 f"MCE: {ece_metrics[1].item():.3f}, "
                 f"BRI: {ece_metrics[2].item():.3f}")

    return best_temp


def offset_client_classes(loader, device):
    for i, batch in enumerate(loader):
        img, label = tuple(t.to(device) for t in batch)
        all_labels = label if i == 0 else torch.cat((all_labels, label))

    client_labels, client_indices = torch.sort(torch.unique(all_labels))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    return label_map


def calc_metrics(results):
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def local_train(args, net, train_loader, pbar, pbar_dict: Dict):
    local_net = copy.deepcopy(net)
    local_net.train()
    optimizer = get_optimizer(args, local_net)
    criteria = torch.nn.CrossEntropyLoss()
    device = get_device()
    train_avg_loss = 0.0
    for i in range(args.inner_steps):
        for k, batch in enumerate(train_loader):
            x, Y = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # forward prop
            pred = local_net(x)
            loss = criteria(pred, Y)

            # back prop
            loss.backward()
            # # clip gradients
            # torch.nn.utils.clip_grad_norm_(local_net.parameters(), args.clip)
            # update local parameters
            optimizer.step()

            # aggregate losses
            train_avg_loss += (loss.item() / Y.shape[0])

            pbar_dict.update({"Inner Step": f'{(i + 1)}'.zfill(3),
                              "Batch": f'{(k + 1)}'.zfill(3),
                              "Train Current Loss": f'{loss.item():5.2f}'})
            pbar.set_postfix(pbar_dict)

        # end of for k, batch in enumerate(train_loader):
    # end of for i in range(args.inner_steps):
    return local_net, train_avg_loss


def get_optimizer(args, network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


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
