import argparse
import os
from pathlib import Path
import torch
from fed_trainers.sweepers.sweep_utils import sweep, load_config
from fed_trainers.trainers.factory import get_logger
from fed_trainers.trainers.dp_sgd import trainer_cifar10_dp_no_gp
from fed_trainers.trainers.params import add_arguments


def main():

    data_name = os.environ.get('DATA_NAME', 'cifar10')
    use_gp = os.environ.get('USE_GP', False)
    dp_method = 'sgd_dp'

    num_classes = 10 if data_name == 'cifar10' else 100
    num_users = 88
    num_public_clients = 6
    working_dir = Path(__file__).resolve().parents[2]

    parser = add_arguments(data_name, dp_method, num_classes, num_public_clients, num_users, use_gp, working_dir, return_parser=True)

    parser.add_argument("--sweep_metric_name", type=str, default="val_avg_acc", help="metric to maximize/minimize in sweep")
    parser.add_argument("--sweep_metric_goal", type=str, default="maximize", choices=['maximize', 'minimize'], help="maximize or minimize in sweep")

    args = parser.parse_args()
    args.wandb = True

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = get_logger(args)
    logger.info(f"Args: {args}")

    # sweep_configuration = {
    #     "name": f"nonoise_SGD_DP_CIFAR10",
    #     # "name": f"SGD_DP_CIFAR10_lr_{args.lr}_seeds{(args.seed, args.seed + 1, args.seed + 2)}",
    #     "method": "grid",
    #     "metric": {"goal": "maximize", "name": "test_acc"},
    #     "parameters": {
    #         "lr": {"values": [1e-2]},
    #         "lr_dec_rate": {"values": [1.0]},
    #         "global_lr": {"values": [1e-1, 1e-2]},
    #         # "min_global_lr": {"values": [0.5, 0.1]},
    #         # "eps": {"values": [8]},
    #         "seed": {"values": [args.seed]},
    #         # "seed": {"values": [args.seed, args.seed + 1, args.seed + 2]},
    #         # "batch_size": {"values": [args.batch_size]},
    #         # "num_public_clients": {"values": [args.num_public_clients]},
    #         "clip": {"values": [10, 1, 1e-1]},
    #         # "calibration_split": {"values": [0.0]},
    #         # "inner_steps": {"values": [1, 3]},
    #         # "wd": {"values": [1e-4]},
    #         "n_epochs": {"values": [10]},
    #         # "optimizer": {"values": ["sgd"]},
    #         # "num_client_agg": {"values": [args.num_client_agg]},
    #         # "model_name": {"values": ["CNNTarget", "ResNet"]},
    #         "noise_multiplier": {"values": [args.noise_multiplier]}
    #     },
    # }

    sweep_name = f"eps{args.eps}_epochs{args.n_epochs}_{dp_method.upper()}_{args.data_name.upper()}_seed{args.seed}"
    if use_gp:
        sweep_name = f"GP_{sweep_name}"
    sweep_configuration = {
        "name": sweep_name,
        "method": "bayes",
        "metric": {"goal": args.sweep_metric_goal, "name": args.sweep_metric_name},
        "parameters": {
            "seed": {"values": [args.seed]},
            "n_epochs": {"min": args.n_epochs, "max": args.n_epochs + 10},
            "num_client_agg": {"values": [args.num_client_agg]},
            "eps": {"values": [args.eps]}
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3, "s": 2, "eta": 3}
    }

    config_path = os.path.join(working_dir, 'sweepers/sweep_configurations/cifar10_sgd_dp_bayes.yaml')
    sweep_configuration['parameters'] =  {**sweep_configuration['parameters'], **load_config(config_path)['parameters']}

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_cifar10_dp_no_gp.train)


if __name__ == '__main__':
    main()
