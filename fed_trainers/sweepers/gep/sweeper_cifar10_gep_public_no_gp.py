import argparse
import logging
from pathlib import Path
import torch
from fed_trainers.trainers.gep import trainer_cifar10_gep_public_no_gp
from fed_trainers.sweepers.sweep_utils import sweep
from fed_trainers.trainers.utils import set_logger, str2bool


def main():
    data_name = 'cifar10'
    dp_method = 'gep_public'
    parser = argparse.ArgumentParser(
        description=f"Sweep GEP Public Federated Learning {data_name.upper()}")
    num_classes = 10 if data_name == 'cifar10' else 100
    num_users = 500
    num_public_clients = 10
    working_dir = Path(__file__).resolve().parents[2]
    ##################################
    #       Network args        #
    ##################################
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=num_classes, help="Number of unique labels")
    parser.add_argument("--model_name", type=str, choices=['CNNTarget', 'ResNet'], default='ResNet')

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='adam',
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
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument("--calibration_split", type=float, default=0.0,
                        help="split ratio of the test set for calibration before testing")
    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp_name", type=str, default=f'Sweep_{dp_method.upper()}_{data_name.upper()}', help="suffix for exp name")
    parser.add_argument("--save_path", type=str, default=(working_dir / 'saved_models').as_posix(),
                        help="dir path for saved models")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=True)

    ##################################
    #       GEP args                 #
    ##################################
    parser.add_argument("--gradients_history_size", type=int,
                        default=500, help="amount of past gradients participating in embedding subspace computation")
    parser.add_argument("--basis_size", type=int, default=50, help="number of basis vectors")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'cifar100', 'putEMG'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data_path", type=str, default=(working_dir / "data").as_posix(), help="dir path for dataset")
    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users - num_public_clients, help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=num_public_clients, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=num_classes // 5, help="number of classes each client experience")

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
    parser.add_argument("--log_name", type=str, default=f"Sweep_{dp_method.upper()}_{data_name.upper()}",
                        help="dir path for logger file")
    parser.add_argument("--csv_path", type=str, default=(working_dir / "csv").as_posix(), help="dir path for csv file")
    parser.add_argument("--csv_name", type=str, default=f"{data_name}_sgd_dp.csv", help="dir path for csv file")


    parser.add_argument("--sweep_metric_name", type=str, default="val_avg_acc", help="metric to maximize/minimize in sweep")
    parser.add_argument("--sweep_metric_goal", type=str, default="maximize", choices=['maximize', 'minimize'], help="maximize or minimize in sweep")

    args = parser.parse_args()



    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = set_logger(args)
    logger.info(f"Args: {args}")

    # sweep_configuration = {
    #     # "name": f"GEP_PUBLIC_CIFAR10_lr_{args.lr}_global_lrs_{(0.9, 0.99, 0.999)}_seed_{args.seed}",
    #     "name": f"GEP_PUBLIC_CIFAR10_epsilon_{args.eps}",
    #     "method": "grid",
    #     "metric": {"goal": "maximize", "name": "test_best_acc"},
    #     "parameters": {
    #         "lr": {"values": [1e-1, 1e-3]},
    #         "global_lr": {"values": [0.99, 0.95]},
    #         # "eps": {"values": [8, 3, 1]},
    #         "seed": {"values": [args.seed]},
    #         # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
    #         "basis_size": {"values": [args.basis_size, args.basis_size / 2]},
    #         "gradients_history_size": {"values": [args.gradients_history_size, args.gradients_history_size / 2]},
    #         # "num_public_clients": {"values": [args.num_public_clients]},
    #         "clip": {"values": [1e-1, 1e-3]},
    #         # "calibration_split": {"values": [0.0]},
    #         # "inner_steps": {"values": [1]},
    #         # "wd": {"values": [1e-4]},
    #         # "n_epochs": {"values": [args.n_epochs]},
    #         # "optimizer": {"values": ["adam"]},
    #         # "num_client_agg": {"values": [10]},
    #         # "model_name": {"values": ["CNNTarget"]},
    #     },
    # }

    sweep_configuration = {
        "name": f"eps{args.eps}_epochs{args.n_epochs}_{dp_method.upper()}_{args.data_name.upper()}_seed{args.seed}",
        # "name": f"SGD_DP_CIFAR10_lr_{args.lr}_seeds{(args.seed, args.seed + 1, args.seed + 2)}",
        "method": "bayes",
        "metric": {"goal": args.sweep_metric_goal, "name": args.sweep_metric_name},
        "parameters": {
            "lr": {"min": 1e-3, "max": 1e-1},
            "lr_dec_rate": {"min": 0.9, "max": 1.0},
            "global_lr": {"min": 0.2, "max": 1.0},
            "seed": {"values": [args.seed]},
            # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
            "basis_size": {"min": args.basis_size // 2, "max": args.basis_size},
            "gradients_history_size": {"min": args.gradients_history_size // 2, "max": args.gradients_history_size},
            # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
            # "batch_size": {"values": [args.batch_size]},
            # "num_public_clients": {"values": [args.num_public_clients]},
            "clip": {"min": 1e-4, "max": 1.0},
            # "calibration_split": {"values": [0.0]},
            # "inner_steps": {"values": [1, 3]},
            "wd": {"min": 1e-4, "max": 1e-3},
            "n_epochs": {"min": args.n_epochs, "max": args.n_epochs + 10},
            # "optimizer": {"values": ["sgd"]},
            # "num_client_agg": {"values": [args.num_client_agg]},
            # "model_name": {"values": ["CNNTarget", "ResNet"]},
            # "noise_multiplier": {"values": [args.noise_multiplier]}
            "eps": {"values": [args.eps]}
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3, "s": 2, "eta": 3}
    }

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_cifar10_gep_public_no_gp.train)


if __name__ == '__main__':
    main()
