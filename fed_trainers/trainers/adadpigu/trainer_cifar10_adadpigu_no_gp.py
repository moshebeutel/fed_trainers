import argparse
import datetime
import os
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from fed_trainers.datasets.dataset import gen_random_loaders
from fed_trainers.trainers.adadpigu.trainer_adadpigu_no_gp import train
from fed_trainers.trainers.utils import set_logger, set_seed, str2bool


def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders




def main():
    parser = argparse.ArgumentParser(description="CIFAR10/100 ADADPIGU Federated Learning")
    data_name = 'cifar10'
    ##################################
    #       Network args             #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--model-name", type=str, choices=['CNN_Tanh', 'CNNTarget', 'ResNet'], default='ResNet')
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--use-gp', type=str2bool, default=False)

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument('--n_epoch', default=500, type=int, help='total number of epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
    # parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=5, help="number of inner steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=0.9, help="server learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")

    parser.add_argument("--calibration_split", type=float, default=0.2,
                        help="split ratio of the test set for calibration before testing")
    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=25, help="eval only after X selected epochs")
    parser.add_argument("--log-every", type=int, default=1, help="log every X selected epochs")
    parser.add_argument('--log_level', default='DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="sgd_dp", help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default=f"{data_name}_sgd_dp.csv", help="dir path for csv file")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'cifar100'], help="dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-classes", type=int, default=10, help="total number of clients")

    #############################
    #       Clients Args        #
    #############################

    parser.add_argument("--num-clients", type=int, default=500, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=490, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=10, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=2, help="number of simulated clients")
    parser.add_argument("--num-client-agg", type=int, default=100, help="number of clients per step")

    parser.add_argument('--private', action='store_true', help='enable differential privacy')
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--noise-multiplier", type=float, default=0.1, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument('--eps', default=4., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--prunnig_rate', default=0.1, type=float, help='aaddpigu prinning rate')

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.debug(f"Args: {args}")
    set_seed(args.seed)

    exp_name = f'SGD-DP_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'

    # Weights & Biases
    if args.wandb:
        wandb.init(project="emg_gp_moshe", name=exp_name)
        wandb.config.update(args)



    print("==> Starting mask-DP training experiments...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = f'results_maskdp_{args.dataset}_eps{args.eps}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)

    trainloaders: tuple[DataLoader, DataLoader, DataLoader] = get_dataloaders(args)


    for batch_size in [args.batch_size]:
    # for batch_size in range(800, 1500 + 1, 100):
    #     args.batchsize = batch_size
        batch_dir = os.path.join(base_dir, f'bs{batch_size}')
        os.makedirs(batch_dir, exist_ok=True)

        for lr in [args.lr]:
        # for lr in [0.05]:
        #     args.lr = round(lr, 2)
            lr_dir = os.path.join(batch_dir, f'lr{args.lr}')
            os.makedirs(lr_dir, exist_ok=True)

            print(f"\n[Run] batch_size={batch_size}, lr={args.lr}")

            for pruning_rate in [0.1]:
                # print(f"\n[Pruning Rate] {pruning_rate}")
                #
                # pruning_rate_dir = os.path.join(lr_dir, f'pruning_rate_{pruning_rate}')
                # os.makedirs(pruning_rate_dir, exist_ok=True)
                #
                # results_file = os.path.join(pruning_rate_dir, 'results.csv')

                train(args, trainloaders)

    print("==> All experiments completed.")


if __name__ == '__main__':
    main()
