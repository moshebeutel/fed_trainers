import argparse
import inspect
from pathlib import Path

import torch
import wandb
from datasets.dataset import gen_random_loaders
from no_gp.trainers.trainer_gep_public_no_gp import train
from common.utils import set_logger, set_seed, str2bool


def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GEP Public CIFAR10/100 Federated Learning")
    data_name = 'cifar10'
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

    ##################################
    #       GEP args                 #
    ##################################
    parser.add_argument("--gradients-history-size", type=int,
                        default=150, help="amount of past gradients participating in embedding subspace computation")
    parser.add_argument("--basis-size", type=int, default=120, help="number of basis vectors")

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
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG', 'mnist'], help="dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=30, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=25, help="number of private clients")
    parser.add_argument("--num-public-clients", type=int, default=5, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=20, help="number of simulated clients")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=25, help="eval only after X selected epochs")

    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="gep_public", help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default=f"{data_name}_sgd_dp.csv", help="dir path for csv file")

    #############################
    #       GP args             #
    #############################

    parser.add_argument('--use-gp', type=str2bool, default=True, help="use gaussian process as "
                                                                      "personalization mechanism")
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
    parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                        choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                        help='kernel function')
    parser.add_argument('--objective', type=str, default='predictive_likelihood',
                        choices=['predictive_likelihood', 'marginal_likelihood'])
    parser.add_argument('--predict-ratio', type=float, default=0.5,
                        help='ratio of samples to make predictions for when using predictive_likelihood objective')
    parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
    parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
    parser.add_argument('--outputscale', type=float, default=8., help='output scale')
    parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
    parser.add_argument('--outputscale-increase', type=str, default='constant',
                        choices=['constant', 'increase', 'decrease'],
                        help='output scale increase/decrease/constant along tree')
    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")
    set_seed(args.seed)

    exp_name = f'GEP_PUBLIC_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'

    # Weights & Biases
    if args.wandb:
        wandb.init(project="key_press_emg_toronto", name=exp_name)
        wandb.config.update(args)

    if args.use_gp:
        from with_gp.trainers import trainer_gep_public_with_gp
        trainers_module = trainer_gep_public_with_gp
        logger.info(f"Using GP trainer: {trainers_module}")
    else:
        from no_gp.trainers import trainer_sgd_dp_no_gp
        trainers_module = trainer_sgd_dp_no_gp
        logger.info(f"Using non-GP trainer: {trainers_module}")

    assert inspect.ismodule(trainers_module), f"trainers_module should be a module"
    assert hasattr(trainers_module, 'train'), f"trainers_module should contain a train function"
    train = getattr(trainers_module, 'train')
    assert inspect.isfunction(train), f"trainers_module.train should be a function"

    train(args, get_dataloaders(args))
