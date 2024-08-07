import argparse
from pathlib import Path

import torch
import wandb
from dataset import gen_random_loaders
from trainer_gep_private_no_gp import train
from utils import set_logger, set_seed, str2bool


def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GEP Private CIFAR10/100 Federated Learning")

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
    parser.add_argument("--num-client-agg", type=int, default=10, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=0.1, help="gradient clip")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument("--basis-gradients-history-size", type=int,
                        default=100, help="amount of past gradients participating in embedding subspace computation")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=False)

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dataset name"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=30, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=30, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=0, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=2, help="number of simulated clients")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=25, help="eval only after X selected epochs")

    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="gep_private", help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default="cifar_sgd_dp.csv", help="dir path for csv file")


    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")
    set_seed(args.seed)

    exp_name = f'GEP_PRIVATE_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'


    # Weights & Biases
    if args.wandb:
        wandb.init(project="key_press_emg_toronto", name=exp_name)
        wandb.config.update(args)

    train(args, get_dataloaders(args))
