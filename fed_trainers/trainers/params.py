import argparse
import time
from argparse import ArgumentError
from pathlib import Path
from fed_trainers.trainers.utils import str2bool


def add_arguments(data_name: str, dp_method: str,
                  num_classes: int,
                  num_public_clients: int,
                  num_users: int, use_gp: str, working_dir: Path):

    parser = argparse.ArgumentParser(
        description=f"{'GP_' if use_gp else ''}{data_name.upper()} {dp_method.upper()} Federated Learning")
    run_tag = f'{data_name}_{dp_method}_{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    parser.add_argument('--run_tag', default=run_tag, type=str, help='run tag')
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--model_name", type=str, choices=['CNNTarget', 'ResNet'], default='ResNet')
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--use-gp', type=str2bool, default=use_gp)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs to train")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num_client_agg", type=int, default=20, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=1.0, help="server learning rate")
    parser.add_argument("--lr_dec_rate", type=float, default=0.99, help="learning rate decrease rate")
    parser.add_argument("--min_global_lr", type=float, default=0.01,
                        help="min value for decreasing server learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    ##################################
    #       DP args                  #
    ##################################
    parser.add_argument("--dp_method", type=str, default=dp_method,
                        choices=['sgd_dp', 'gep_public'], help="Differential Privacy method")
    parser.add_argument("--clip", type=float, default=1, help="gradient clip")
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
    parser.add_argument("--exp_name", type=str, default=f'{dp_method.upper()}_{data_name.upper()}',
                        help="suffix for exp name")
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
    parser.add_argument("--log_dir", type=str, default=(working_dir / "log").as_posix(),
                        help="dir path for logger file")
    parser.add_argument("--log_name", type=str, default=f'{dp_method}_{data_name}', help="dir path for logger file")
    parser.add_argument("--csv_path", type=str, default=(working_dir / 'csv').as_posix(), help="dir path for csv file")
    parser.add_argument("--csv_name", type=str, default=f"{data_name}_{dp_method}.csv", help="dir path for csv file")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG', 'mnist'], help="dataset"
    )

    parser.add_argument("--num_classes", type=int, default=num_classes, help="total number of clients")

    #############################
    #       Clients Args        #
    #############################

    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users - num_public_clients,
                        help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=num_public_clients, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=num_classes,
                        help="number of classes each client knows")

    if use_gp:
        from fed_trainers.trainers import gp_utils
        parser = gp_utils.add_arguments_gp(parser)

    if 'gep_' in dp_method.lower():
        from fed_trainers.trainers.gep import gep_utils
        parser = gep_utils.add_arguments_gep(parser)

    if 'emg' in data_name.lower():
        if data_name == 'putEMG':
            from fed_trainers.datasets.emg_utils import add_arguments_putemg
            parser = add_arguments_putemg(parser, working_dir)
        elif data_name == 'keypressemg':
            from fed_trainers.datasets.keypressemg.keypressemg_utils import add_arguments_keypressemg
            parser = add_arguments_keypressemg(parser, working_dir)
        else:
            raise ArgumentError(f'{data_name} not supported')
    else:
        parser.add_argument("--data_path", type=str, default=(working_dir / f"data/{data_name.upper()}/").as_posix(),
                            help="dir path for dataset")


    args = parser.parse_args()
    return args
