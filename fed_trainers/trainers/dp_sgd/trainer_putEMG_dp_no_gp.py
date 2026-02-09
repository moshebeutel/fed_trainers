import argparse
import os
import time
from pathlib import Path
import torch
import wandb
from fed_trainers.datasets.emg_utils import get_dataloaders, get_num_users
from fed_trainers.trainers import gp_utils
from fed_trainers.trainers.utils import set_logger, set_seed, str2bool, log_data_statistics, compute_sample_probability, \
    compute_steps, get_sigma


def train(args):
    set_seed(args.seed)
    dataloaders = get_dataloaders(args)
    log_data_statistics(dataloaders, args)

    q = compute_sample_probability(args)
    steps = compute_steps(args)
    logger = set_logger(args)
    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps, args.eps, args.delta, rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    if args.use_gp:
        from fed_trainers.trainers.dp_sgd import trainer_sgd_dp_with_gp as trainer
    else:
        from fed_trainers.trainers.dp_sgd import trainer_sgd_dp_no_gp as trainer

    trainer.train(args, dataloaders)

def main():

    data_name = 'putEMG'
    use_gp = os.environ.get('USE_GP', 'False')
    dp_method = 'sgd_dp'
    parser = argparse.ArgumentParser(
        description=f"{'GP_' if use_gp else ''}{data_name.upper()} {dp_method.upper()} Federated Learning")
    num_users = get_num_users() * 2
    num_classes = 4
    num_public_clients = 6
    working_dir = Path(__file__).resolve().parents[2]
    run_tag = f'{data_name}_{dp_method}_{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    parser.add_argument('--run_tag', default=run_tag, type=str, help='run tag')
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--model_name", type=str, choices=['FeatureModel', 'ResNet'], default='FeatureModel')
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=num_classes, help="Number of unique labels")
    parser.add_argument("--num-features", type=int, default=384, help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=16, help="Number of extracted features per channel")
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--use-gp', type=str2bool, default=use_gp)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num_client_agg", type=int, default=20, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=1.0, help="server learning rate")
    parser.add_argument("--lr_dec_rate", type=float, default=0.95, help="learning rate decrease rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--noise_multiplier", type=float, default=0.0, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument("--calibration_split", type=float, default=0.0,
                        help="split ratio of the test set for calibration before testing")
    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp_name", type=str, default=f'{dp_method.upper()}_{data_name.upper()}', help="suffix for exp name")
    parser.add_argument("--save_path", type=str, default=(working_dir / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval_every", type=int, default=1, help="eval every X selected epochs")
    parser.add_argument("--eval_after", type=int, default=0, help="eval only after X selected epochs")
    parser.add_argument("--log_every", type=int, default=1, help="log every X selected epochs")
    parser.add_argument('--log_level', default='DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument("--log_dir", type=str, default=(working_dir  / "log").as_posix(), help="dir path for logger file")
    parser.add_argument("--log_name", type=str, default=f'{dp_method}_{data_name}', help="dir path for logger file")
    parser.add_argument("--csv_path", type=str, default=(working_dir / 'csv').as_posix(), help="dir path for csv file")
    parser.add_argument("--csv_name", type=str, default=f"{data_name}_{dp_method}.csv", help="dir path for csv file")
    parser.add_argument('--log-data-statistics', type=str2bool, default=False)


    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data_name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG'], help="dataset name"
    )
    parser.add_argument("--data_path", type=str,
                        # default='./data/EMG/putEMG/Data-HDF5-Features-NoArgs',
                        default=(working_dir / 'data/EMG/putEMG/Data-HDF5-Features-Short-Time').as_posix(),
                        # default='./data/EMG/putEMG/Data-HDF5-Features-Small',
                        # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                        help="dir path for dataset")
    parser.add_argument("--num_classes", type=int, default=num_classes, help="total number of clients")

    #############################
    #       Clients Args        #
    #############################

    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users-num_public_clients, help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=num_public_clients, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=num_classes,
                        help="number of classes each client knows")



    if use_gp:
        parser = gp_utils.parse_args(parser)
    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")
    logger.debug('Debug Logger Set')

    set_seed(args.seed)

    q = compute_sample_probability(args)
    steps = compute_steps(args)

    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps, args.eps, args.delta, rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    exp_name = f'{dp_method.upper()}_{data_name.upper()}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}_seed_{args.seed}'
    if use_gp:
        exp_name = f'GP_{exp_name}'

    # Weights & Biases
    if args.wandb:
        run = wandb.init(project="dec25_sweeps", name=exp_name, tags=[args.run_tag])
        wandb.config.update(args)

    train(args)

    if args.wandb:
        run.finish()

if __name__ == '__main__':
    main()
