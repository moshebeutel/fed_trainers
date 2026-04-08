import os
from pathlib import Path
import torch
import wandb

from fed_trainers.datasets.keypressemg.keypressemg_utils import get_num_users
from fed_trainers.trainers.factory import get_dataloaders, get_trainer, get_logger
from fed_trainers.trainers.params import add_arguments
from fed_trainers.trainers.utils import set_seed, log_data_statistics, compute_sample_probability, \
    compute_steps, get_sigma, create_wandb_report


def train(args):
    set_seed(args.seed)
    dataloaders = get_dataloaders(args)
    log_data_statistics(dataloaders, args)
    q = compute_sample_probability(args)
    steps = compute_steps(args)
    logger = get_logger(args)
    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps,
                                                                                                         args.eps,
                                                                                                         args.delta,
                                                                                                         rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    trainer = get_trainer(args)

    trainer.train(args, dataloaders)

def main():

    data_name = os.environ.get('DATA_NAME', 'keypressemg')
    use_gp = os.environ.get('USE_GP', 'False')
    dp_method = "sgd_dp"
    num_users = get_num_users()
    num_classes = 26
    num_public_clients = 3
    working_dir = Path(__file__).resolve().parents[2]

    args = add_arguments(data_name, dp_method, num_classes, num_public_clients, num_users, use_gp, working_dir)

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = get_logger(args)
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
        report = create_wandb_report(args)


    train(args)

    if args.wandb:
        run.finish()

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description="Toronto Surface EMG Typing Database SGD-DP Federated Learning")
#     num_users = get_num_users()
#     ##################################
#     #       Network args        #
#     ##################################
#     parser.add_argument("--depth_power", type=int, default=1)
#     parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")
#     parser.add_argument("--num-features", type=int, default=320, help="Number of extracted features (model input size)")
#     parser.add_argument("--num-features-per-channel", type=int, default=20,
#                         help="Number of extracted features per channel")
#
#     ##################################
#     #       Optimization args        #
#     ##################################
#     parser.add_argument("--num-steps", type=int, default=200)
#     parser.add_argument("--optimizer", type=str, default='sgd',
#                         choices=['adam', 'sgd'], help="optimizer type")
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--inner-steps", type=int, default=5, help="number of inner steps")
#     parser.add_argument("--num-client-agg", type=int, default=num_users, help="number of clients per step")
#     parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
#     parser.add_argument("--global_lr", type=float, default=0.999, help="server learning rate")
#     parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
#     parser.add_argument("--clip", type=float, default=10.0, help="gradient clip")
#     parser.add_argument("--noise-multiplier", type=float, default=0.0, help="dp noise factor "
#                                                                             "to be multiplied by clip")
#     parser.add_argument("--calibration_split", type=float, default=0.0,
#                         help="split ratio of the test set for calibration before testing")
#
#     #############################
#     #       General args        #
#     #############################
#     parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
#     parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
#     parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
#     parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
#                         help="dir path for saved models")
#     parser.add_argument("--seed", type=int, default=42, help="seed value")
#     parser.add_argument('--wandb', type=str2bool, default=False)
#     parser.add_argument('--log_data_statistics', type=str2bool, default=False)
#
#     #############################
#     #       Dataset Args        #
#     #############################
#
#     parser.add_argument(
#         "--data-name", type=str, default="keypressemg",
#         choices=['cifar10', 'cifar100', 'putEMG', 'keypressemg'], help="Name of the dataset"
#     )
#     parser.add_argument("--data-path", type=str,
#                         default='./data/EMG/keypressemg/CleanData/valid_features_long_npy',
#                         # default=(Path.cwd() / 'data/valid_user_features').as_posix(),
#                         # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
#                         help="dir path for dataset")
#     parser.add_argument("--num-clients", type=int, default=num_users, help="total number of clients")
#     parser.add_argument("--num-private-clients", type=int, default=num_users, help="number of private clients")
#     parser.add_argument("--num-public-clients", type=int, default=0, help="number of public clients")
#     parser.add_argument("--classes-per-client", type=int, default=26, help="number of classes each client experience")
#
#     #############################
#     #       General args        #
#     #############################
#     parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
#     parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
#     parser.add_argument("--eval-after", type=int, default=1, help="eval only after X selected epochs")
#
#     parser.add_argument("--log-every", type=int, default=5, help="log every X selected epochs")
#     parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
#     parser.add_argument("--log-name", type=str, default="sgd_dp_keypressemg", help="dir path for logger file")
#     parser.add_argument("--log-level", type=int, default=logging.INFO, help="logger filter")
#     parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
#     parser.add_argument("--csv-name", type=str, default="keypressemg_sgd_dp.csv", help="dir path for csv file")
#     parser.add_argument("--distance_matrix_file", type=str,
#                         default="data/Alphabetically_Sorted_QWERTY_Distance_Matrix.csv", help="dir path for csv file")
#
#     args = parser.parse_args()
#
#     assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
#
#     logger = get_logger(args)
#     logger.info(f"Args: {args}")
#
#     exp_name = f'SGD-DP_{args.data_name}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}'
#
#     # Weights & Biases
#     if args.wandb:
#         wandb.init(project="emg_gp_moshe", name=exp_name)
#         wandb.config.update(args)
#
#     train(args)
