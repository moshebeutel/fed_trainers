import argparse
import logging
from pathlib import Path
import torch
import trainer_putEMG_gep_public_no_gp
from emg_utils import get_num_users
from sweep_utils import sweep
from utils import set_logger, str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sweep GEP Public Federated Learning putEMG Hand Gestures Database")
    num_users = get_num_users()
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=8, help="Number of unique labels")
    parser.add_argument("--num-features", type=int, default=888, help="Number of extracted features (model input size)")

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num-client-agg", type=int, default=5, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=0.1, help="server learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=10.0, help="gradient clip")
    parser.add_argument("--noise_multiplier", type=float, default=0.1, help="dp noise factor "
                                                                            "to be multiplied by clip")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='Sweep_GEP_PUBLIC_putEMG', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=True)

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
        "--data-name", type=str, default="putEMG",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str,
                        default='./data/EMG/putEMG/Data-HDF5-Features-Short-Time',
                        # default='./data/EMG/putEMG/Data-HDF5-Features-Small',
                        # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                        help="dir path for dataset")
    parser.add_argument("--num-clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num-private-clients", type=int, default=num_users - 5, help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=5, help="number of public clients")
    parser.add_argument("--classes-per-client", type=int, default=8, help="number of classes each client experience")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=4, help="eval only after X selected epochs")

    parser.add_argument("--log-every", type=int, default=5, help="log every X selected epochs")
    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="logger filter")
    parser.add_argument("--log-name", type=str, default="Sweep_GEP_PUBLIC_putEMG",
                        help="dir path for logger file")
    parser.add_argument("--csv-path", type=str, default="./csv", help="dir path for csv file")
    parser.add_argument("--csv-name", type=str, default="putemg_gep_public.csv", help="dir path for csv file")

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")

    sweep_configuration = {
        "name": f"gep_public_putEMG_{args.num_features}_{args.seed}",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_avg_acc"},
        "parameters": {
            "lr": {"values": [0.01]},
            # "lr": {"values": [0.001, 0.01]},
            "global_lr": {"values": [0.999]},
            # "global_lr": {"values": [0.9, 0.1]},
            "seed": {"values": [args.seed]},
            # "clip": {"values": [1.0]},
            "clip": {"values": [10.0, 0.01]},
            "noise_multiplier": {"values": [0.0, 0.1, 1.0, 10.0]},
            "inner_steps": {"values": [10.0]},
            # "inner_steps": {"values": [1.0, 10.0]},
            "basis-size": {"values": [10]},
            "gradients-history-size": {"values": [20]},
            "wd": {"values": [0.0001]},
            # "wd": {"values": [0.0001, 0.001]},
            "num_steps": {"values": [50]},
            "num_client_agg": {"values": [num_users]},
            "depth_power": {"values": [1]}
        },
    }
    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_putEMG_gep_public_no_gp.train)
