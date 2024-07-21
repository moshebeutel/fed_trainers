import argparse
import copy
import logging
from collections import OrderedDict
from typing import Dict
import numpy as np
import torch
import wandb
from sklearn import metrics
from tqdm import trange
from dataset import gen_random_loaders
from model import ResNet
from utils import get_device, set_logger, set_seed, str2bool, initialize_weights


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
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders


def get_model(args):
    num_classes = {'cifar10': 10, 'cifar100': 100, 'putEMG': 8}[args.data_name]
    model = ResNet(layers=[args.block_size] * args.num_blocks, num_classes=num_classes)
    initialize_weights(model)
    return model


@torch.no_grad()
def get_dp_noise(args, net):
    noises = {}
    for n, p in net.named_parameters():
        noise = torch.normal(mean=0.0, std=args.noise_multiplier * args.clip,
                             size=(args.num_steps, args.num_client_agg, *p.shape))
        noises[n] = noise
    return noises


@torch.no_grad()
def eval_model(args, global_model, client_ids, loaders):
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

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

    fields_list = ["num_blocks", "block_size", "optimizer", "lr", "num_client_agg", "clip", "noise_multiplier"]
    args_list = [(k, vars(args)[k]) for k in fields_list]

    logging.info(f' *** Training for args {args_list} ***')

    val_avg_loss, val_avg_acc, val_avg_acc_score, val_avg_f1 = 0.0, 0.0, 0.0, 0.0
    val_acc_dict, val_loss_dict, val_acc_score_dict, val_f1s_dict = {}, {}, {}, {}
    public_clients, private_clients, dummy_clients = get_clients(args)
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    net = get_model(args)
    net = net.to(device)
    best_model = copy.deepcopy(net)
    criteria = torch.nn.CrossEntropyLoss()

    dp_noise_dict = get_dp_noise(args, net)
    dp_noise_dict = {n: noise.to(device) for n, noise in dp_noise_dict.items()}

    train_loaders, val_loaders, test_loaders = get_dataloaders(args)

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0, 0., 0., 0.
    step_iter = trange(args.num_steps)

    for step in step_iter:

        # select several clients
        client_ids_step = np.random.choice(private_clients, size=args.num_client_agg, replace=False)

        # initialize global model params
        params = OrderedDict()
        for n, p in net.named_parameters():
            params[n] = torch.zeros_like(p.data)

        # iterate over each client
        train_avg_loss = 0

        for j, c_id in enumerate(client_ids_step):

            local_net = copy.deepcopy(net)
            local_net.train()
            optimizer = get_optimizer(args, local_net)

            train_loader = train_loaders[c_id]

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
                        f"Step: {step + 1:2d},"
                        f"client: {c_id:2d} ({j + 1} in step),"
                        f"Inner Step: {i + 1:2d},"
                        f"batch: {k + 1:2d},"
                        f"Loss: {loss.item() : 5.2f},"
                        f"train loss {train_avg_loss: 5.2f},"
                        f"best epoch: {best_epoch + 1: 2d},"
                        f"test avg acc: {val_avg_acc:.4f},"
                        f"best test acc: {best_acc:.2f}"
                    )
            # get client parameters and sum.
            for n, p in local_net.named_parameters():
                params[n] += p.data + dp_noise_dict[n][step, j]

        # average parameters
        for n, p in params.items():
            params[n] = p / args.num_client_agg

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
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num-client-agg", type=int, default=50, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--noise-multiplier", type=float, default=0.0, help="dp noise factor "
                                                                            "to be multiplied by clip")

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
        "--data-name", type=str, default="cifar10",
        choices=['cifar10', 'cifar100'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=50, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=50, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=0, help="number of public clients")
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
