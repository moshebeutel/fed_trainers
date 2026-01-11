import argparse
import logging
from pathlib import Path
import torch
from fed_trainers.trainers.dp_sgd import trainer_cifar10_dp_no_gp
from fed_trainers.sweepers.sweep_utils import sweep
from fed_trainers.trainers.utils import set_logger, str2bool


def main():
    parser = argparse.ArgumentParser(
        description="Sweep SGD_DP Federated Learning CIFAR10")
    num_users = 500
    num_public_clients = 10
    working_dir = Path(__file__).resolve().parents[2]
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10, help="Number of unique labels")
    parser.add_argument("--model_name", type=str, choices=['CNNTarget', 'ResNet'], default='ResNet')

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=1, help="number of inner steps")
    parser.add_argument("--num_client_agg", type=int, default=50, help="number of clients per step")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--global_lr", type=float, default=1.0, help="server learning rate")
    parser.add_argument("--lr_dec_rate", type=float, default=0.75, help="learning rate decrease rate")
    parser.add_argument("--min_global_lr", type=float, default=0.01,
                        help="min value for decreasing server learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip", type=float, default=1, help="gradient clip")
    parser.add_argument("--noise_multiplier", type=float, default=0.0, help="dp noise factor "
                                                                            "to be multiplied by clip")
    parser.add_argument('--eps', default=-1, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument("--calibration_split", type=float, default=0.0,
                        help="split ratio of the test set for calibration before testing")
    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp_name", type=str, default='Sweep_SGD_DP_CIFAR10', help="suffix for exp name")
    parser.add_argument("--save_path", type=str, default=(working_dir / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=True)


    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data_path", type=str, default=(working_dir / "data").as_posix(), help="dir path for dataset")
    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users - num_public_clients, help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=num_public_clients, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=2, help="number of classes each client experience")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval_every", type=int, default=1, help="eval every X selected epochs")
    parser.add_argument("--eval_after", type=int, default=1, help="eval only after X selected epochs")

    parser.add_argument("--log_every", type=int, default=1, help="log every X selected epochs")
    parser.add_argument("--log_dir", type=str, default=(working_dir / "log").as_posix(),
                        help="dir path for logger file")
    parser.add_argument("--log_level", type=int, default=logging.INFO, help="logger filter")
    parser.add_argument("--log_name", type=str, default="Sweep_SGD_DP_CIFAR10",
                        help="dir path for logger file")
    parser.add_argument("--csv_path", type=str, default=(working_dir / "csv").as_posix(), help="dir path for csv file")
    parser.add_argument("--csv_name", type=str, default="cifar10_sgd_dp.csv", help="dir path for csv file")

    args = parser.parse_args()

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")

    sweep_configuration = {
        "name": f"agg{args.num_client_agg}_minglr{args.min_global_lr}_SGD_DP_CIFAR10_epsilon_{args.eps}",
        # "name": f"SGD_DP_CIFAR10_lr_{args.lr}_seeds{(args.seed, args.seed + 1, args.seed + 2)}",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_acc"},
        "parameters": {
            "lr": {"values": [1e-3]},
            "lr_dec_rate": {"values": [1.0, 0.9]},
            # "global_lr": {"values": [0.999, 0.9]},
            # "min_global_lr": {"values": [0.5, 0.1]},
            # "eps": {"values": [8]},
            "seed": {"values": [args.seed]},
            # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
            # "batch_size": {"values": [args.batch_size]},
            # "num_public_clients": {"values": [args.num_public_clients]},
            # "clip": {"values": [1e-4, 1]},
            # "calibration_split": {"values": [0.0]},
            # "inner_steps": {"values": [1, 3]},
            # "wd": {"values": [1e-4]},
            # "n_epochs": {"values": [50]},
            # "optimizer": {"values": ["sgd"]},
            # "num_client_agg": {"values": [args.num_client_agg]},
            # "model_name": {"values": ["CNNTarget", "ResNet"]},
        },
    }
    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_cifar10_dp_no_gp.train)


if __name__ == '__main__':
    main()
