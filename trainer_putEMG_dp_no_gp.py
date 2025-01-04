import argparse
from pathlib import Path
import torch
import wandb
from emg_utils import get_dataloaders, get_user_list
from utils import set_logger, set_seed, str2bool
from trainer_sgd_dp_no_gp import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="putEMG SGD-DP Federated Learning")
    num_users = len(get_user_list())
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
    parser.add_argument("--num-client-agg", type=int, default=5, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=0.9, help="server learning rate")
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
    parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=False)

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="putEMG",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str,
                        default='./data/EMG/putEMG/Data-HDF5-Features-Small',
                        # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                        help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=num_users, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=0, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=2, help="number of simulated clients")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=25, help="eval only after X selected epochs")

    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="sgd_dp_emg", help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default="emg_sgd_dp.csv", help="dir path for csv file")


    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")
    set_seed(args.seed)

    exp_name = f'SGD-DP_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'

    # Weights & Biases
    if args.wandb:
        wandb.init(project="emg_gp_moshe", name=exp_name)
        wandb.config.update(args)

    train(args, get_dataloaders(args))
