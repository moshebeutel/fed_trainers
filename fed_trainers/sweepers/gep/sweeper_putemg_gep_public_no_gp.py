import argparse
import logging
from pathlib import Path
import torch
from fed_trainers.trainers.gep import trainer_putEMG_gep_public_no_gp
from fed_trainers.datasets.emg_utils import get_num_users
from fed_trainers.sweepers.sweep_utils import sweep
from fed_trainers.trainers.utils import set_logger, str2bool


def main():
    data_name = 'putEMG'
    dp_method = 'gep_public'
    parser = argparse.ArgumentParser(description=f"Sweep {dp_method.upper()} {data_name} Federated Learning")
    num_users = get_num_users()
    num_classes = 4
    num_public_clients = 5
    working_dir = Path(__file__).resolve().parents[2]
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--model_name", type=str, choices=['FeatureModel', 'ResNet'], default='FeatureModel')
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=num_classes, help="Number of unique labels")
    parser.add_argument("--num-features", type=int, default=384, help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=16, help="Number of extracted features per channel")


    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--inner_steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num_client_agg", type=int, default=10, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=1.0, help="server learning rate")
    parser.add_argument("--lr_dec_rate", type=float, default=0.75, help="learning rate decrease rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=10.0, help="gradient clip")
    parser.add_argument("--noise_multiplier", type=float, default=0.0, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument('--eps', default=8, type=float, help='privacy parameter epsilon')
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
    parser.add_argument('--wandb', type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval_every", type=int, default=1, help="eval every X selected epochs")
    parser.add_argument("--eval_after", type=int, default=0, help="eval only after X selected epochs")
    parser.add_argument("--log_every", type=int, default=1, help="log every X selected epochs")
    parser.add_argument('--log_level', default='INFO', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument("--log-dir", type=str, default=(working_dir  / "log").as_posix(), help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default=f"{data_name}_{dp_method}", help="dir path for logger file")
    parser.add_argument("--csv_path", type=str, default=(working_dir / 'csv').as_posix(), help="dir path for csv file")
    parser.add_argument("--csv_name", type=str, default=f"{data_name}_{dp_method}.csv", help="dir path for csv file")
    parser.add_argument('--log-data-statistics', type=str2bool, default=False)

    ##################################
    #       GEP args                 #
    ##################################
    parser.add_argument("--gradients-history-size", type=int,
                        default=100, help="amount of past gradients participating in embedding subspace computation")
    parser.add_argument("--basis-size", type=int, default=40, help="number of basis vectors")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data_name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG'], help="dataset name"
    )
    parser.add_argument("--data_path", type=str,
                        # default='./data/EMG/putEMG/Data-HDF5-Features-NoArgs',
                        default='./data/EMG/putEMG/Data-HDF5-Features-Short-Time',
                        # default='./data/EMG/putEMG/Data-HDF5-Features-Small',
                        # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                        help="dir path for dataset")

    #############################
    #       Clients Args        #
    #############################

    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users - num_public_clients,
                        help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=num_public_clients, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=num_classes,
                        help="number of classes each client knows")


    parser.add_argument("--sweep_metric_name", type=str, default="val_avg_acc", help="metric to maximize/minimize in sweep")
    parser.add_argument("--sweep_metric_goal", type=str, default="maximize", choices=['maximize', 'minimize'], help="maximize or minimize in sweep")

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")

    # sweep_configuration = {
    #     "name": f"gep_public_putEMG_{args.num_features}_{args.seed}",
    #     "method": "grid",
    #     "metric": {"goal": "maximize", "name": "test_best_acc"},
    #     "parameters": {
    #         "lr": {"values": [0.1]},
    #         "global_lr": {"values": [0.999, 0.5]},
    #         "seed": {"values": [args.seed]},
    #         "clip": {"values": [10.0, 1.0, 0.1, 0.01]},
    #         "noise_multiplier": {"values": [0.0, 0.1, 1.0, 10.0]},
    #         "inner_steps": {"values": [1]},
    #         "basis-size": {"values": [19]},
    #         "gradients-history-size": {"values": [20]},
    #         "num_public_clients": {"values": [5]},
    #         "wd": {"values": [0.0001, 0.001]},
    #         "num_steps": {"values": [100]},
    #         "num_client_agg": {"values": [5]},
    #         "depth_power": {"values": [1]}
    #     },
    # }

    sweep_configuration = {
        "name": f"eps{args.eps}_epochs{args.n_epochs}_{dp_method.upper()}_{args.data_name.upper()}_seed{args.seed}",
        "method": "bayes",
        "metric": {"goal": args.sweep_metric_goal, "name": args.sweep_metric_name},
        "parameters": {
            "lr": {"min": 1e-2, "max": 1e-1},
            "lr_dec_rate": {"min": 0.9, "max": 1.0},
            "global_lr": {"min": 0.1, "max": 1.0},
            "seed": {"values": [args.seed]},
            # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
            "basis_size": {"min": args.basis_size // 2, "max": args.basis_size},
            "gradients_history_size": {"min": args.gradients_history_size // 2, "max": args.gradients_history_size},
            "batch_size": {"values": [args.batch_size, args.batch_size*2]},
            "clip": {"min": 1e-4, "max": 1.0},
            # "calibration_split": {"values": [0.0]},
            # "inner_steps": {"values": [1, 3]},
            "wd": {"min": 1e-4, "max": 1e-3},
            "n_epochs": {"min": args.n_epochs, "max": args.n_epochs + 20},
            # "optimizer": {"values": ["sgd"]},
            # "num_client_agg": {"values": [args.num_client_agg]},
            # "noise_multiplier": {"values": [args.noise_multiplier]}
            "eps": {"values": [args.eps]}
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3, "s": 2, "eta": 3}
    }

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_putEMG_gep_public_no_gp.train)
