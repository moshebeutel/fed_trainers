import argparse
from pathlib import Path

import torch
import wandb
from fed_trainers.datasets.dataset import gen_random_loaders
from fed_trainers.trainers.gep.trainer_gep_private_no_gp import train
from fed_trainers.trainers.utils import set_logger, set_seed, str2bool, compute_steps, \
    compute_sample_probability, get_sigma


def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders


def main():
    parser = argparse.ArgumentParser(description="GEP Private CIFAR10/100 Federated Learning")

    data_name = 'cifar10'

    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--model-name", type=str, choices=['CNNTarget', 'ResNet'], default='ResNet')

    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--use-gp', type=str2bool, default=False)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default='adam',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=15, help="number of inner steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=0.9, help="server learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=0.1, help="gradient clip")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument("--calibration_split", type=float, default=0.2,
                        help="split ratio of the test set for calibration before testing")

    ##################################
    #       GEP args                 #
    ##################################
    parser.add_argument("--gradients-history-size", type=int,
                        default=500, help="amount of past gradients participating in embedding subspace computation")
    parser.add_argument("--basis-size", type=int, default=50, help="number of basis vectors")

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
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=1, help="eval only after X selected epochs")
    parser.add_argument("--log-every", type=int, default=1, help="log every X selected epochs")
    parser.add_argument('--log_level', default='INFO', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="gep_private", help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default=f"{data_name}_sgd_dp.csv", help="dir path for csv file")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG', 'mnist'], help="dataset"
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

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")
    set_seed(args.seed)

    # trainloaders: tuple[DataLoader, DataLoader, DataLoader] = get_dataloaders(args)
    #
    # n_training = len(trainloaders[0].dataset)
    q = compute_sample_probability(args)
    steps = compute_steps(args)

    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps, args.eps, args.delta, rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    exp_name = f'GEP_PRIVATE_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'


    # Weights & Biases
    if args.wandb:
        wandb.init(project="key_press_emg_toronto", name=exp_name)
        wandb.config.update(args)

    train(args, get_dataloaders(args))

if __name__ == '__main__':
    main()
