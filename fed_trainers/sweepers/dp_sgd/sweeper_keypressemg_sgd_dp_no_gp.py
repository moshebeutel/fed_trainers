import os
from pathlib import Path
import torch
from fed_trainers.trainers.dp_sgd import trainer_keypressemg_sgd_dp_no_gp
from fed_trainers.datasets.keypressemg.keypressemg_utils import get_num_users
from fed_trainers.sweepers.sweep_utils import sweep, load_config
from fed_trainers.trainers.factory import get_logger
from fed_trainers.trainers.params import add_arguments

def main():

    data_name = os.environ.get('DATA_NAME', 'keypressemg')
    use_gp = os.environ.get('USE_GP', False)
    dp_method = 'sgd_dp'

    num_classes = 26
    num_users = get_num_users()
    num_public_clients = 3
    working_dir = Path(__file__).resolve().parents[2]

    parser = add_arguments(data_name, dp_method, num_classes, num_public_clients, num_users, use_gp, working_dir, return_parser=True)

    parser.add_argument("--sweep_metric_name", type=str, default="val_avg_acc", help="metric to maximize/minimize in sweep")
    parser.add_argument("--sweep_metric_goal", type=str, default="maximize", choices=['maximize', 'minimize'], help="maximize or minimize in sweep")


    args = parser.parse_args()
    args.wandb = True
    args.log_level = 'INFO'

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = get_logger(args)
    logger.info(f"Args: {args}")

    # sweep_configuration = {
    #     "name": f"sgd_dp_keypressemg_{args.num_features}_{103}",
    #     # "name": f"sgd_dp_keypressemg_{args.num_features}_{103_110}",
    #     "method": "grid",
    #     "metric": {"goal": "maximize", "name": "best_acc"},
    #     "parameters": {
    #         "lr": {"values": [0.01]},
    #         # "lr": {"values": [0.1]},
    #         "global_lr": {"values": [0.999]},
    #         "seed": {"values": [103, 104, 105]},
    #         # "seed": {"values": [103, 104, 105, 106, 107, 108, 109, 110]},
    #         "clip": {"values": [10.0, 20.0]},
    #         # "clip": {"values": [10.0, 1.0, 0.1, 0.01]},
    #         "noise_multiplier": {"values": [0.0]},
    #         # "noise_multiplier": {"values": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]},
    #         "calibration_split": {"values": [0.0]},
    #         # "calibration_split": {"values": [0.0, 0.1, 0.2]},
    #         "inner_steps": {"values": [20]},
    #         "num_steps": {"values": [400]},
    #         "wd": {"values": [0.001]},
    #         "num_client_agg": {"values": [5]},
    #         "depth_power": {"values": [1]},
    #         "log_data_statistics": {"values": [False]}
    #     },
    # }
    # sweep_name = f"eps{args.eps}_epochs{args.n_epochs}_{dp_method.upper()}_{args.data_name.upper()}_seed{args.seed}"
    # if use_gp:
    #     sweep_name = f"GP_{sweep_name}"
    # sweep_configuration = {
    #     "name": sweep_name,
    #     "method": "bayes",
    #     "metric": {"goal": args.sweep_metric_goal, "name": args.sweep_metric_name},
    #     "parameters": {
    #         "seed": {"values": [args.seed]},
    #         "n_epochs": {"min": args.n_epochs, "max": args.n_epochs + 10},
    #         "num_client_agg": {"values": [args.num_client_agg]},
    #         "eps": {"values": [args.eps]}
    #     },
    #     "early_terminate": {"type": "hyperband", "min_iter": 3, "s": 2, "eta": 3}
    # }
    #
    # config_path = os.path.join(working_dir, 'sweepers/sweep_configurations/keypressemg_sgd_dp_bayes.yaml')
    sweep_name = f"FED_EPS_{args.eps}_{dp_method.upper()}_{args.data_name.upper()}"
    if use_gp:
        sweep_name = f"GP_{sweep_name}"
    sweep_configuration = {
        "name": sweep_name,
        "method": "grid",
        "metric": {"goal": args.sweep_metric_goal, "name": args.sweep_metric_name},
        "parameters": {
            "num_client_agg": {"values": [args.num_client_agg]},
            "eps": {"values": [args.eps]}
        },
    }

    config_path = os.path.join(working_dir, 'sweepers/sweep_configurations/keypressemg_sgd_dp_grid.yaml')

    sweep_configuration['parameters'] = {**sweep_configuration['parameters'], **load_config(config_path)['parameters']}

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_keypressemg_sgd_dp_no_gp.train)

if __name__ == '__main__':
    main()