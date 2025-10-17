# import math
# import subprocess

# # for data_name in ['putEMG', 'cifar10', 'mnist']:
# for data_name in ['femnist']:
#
#     print(f'@@@ *** %%% GEP_PUBLIC  {data_name} %%% *** @@@')
#
#     num_users = 500
#     # data_name = 'mnist'
#     classes_per_client = 2 if data_name in ['cifar10', 'mnist'] else 20
#     script_name = 'cifar10'
#     public_client_num_list = [10]
#     if data_name == 'putEMG':
#         script_name = 'putEMG'
#         num_users = 44
#         classes_per_client = 8
#         public_client_num_list = [5]
#     elif data_name == 'femnist':
#         script_name = 'femnist'
#         num_users = 3597
#         classes_per_client = 62
#         public_client_num_list = [100]
#     for num_epochs in [2]:
#     # for num_epochs in [3]:
#         for num_clients in [num_users]:
#             for num_client_agg in [5]:
#                 for num_blocks in [3]:
#                     for block_size in [1]:
#                         for sigma in [0.0, 2.016, 4.72, 12.79, 25.0]:
#                         #     for optimizer in ['adam', 'sgd']:
#                             for optimizer in ['adam']:
#                                 for lr in [0.001]:
#                                 # for lr in [0.01, 0.001]:
#                                     for num_public_clients in public_client_num_list:
#                                         for history_size in [200]:
#                                         # for history_size in [160]:
#                                             # for basis_size in [25, 50]:
#                                             for basis_size in [num_public_clients]:
#                                                 clip_list = [5.0, 1.0] if sigma == 0.0 else [0.001, 0.01]
#                                                 for grad_clip in clip_list:
#                                                 # for grad_clip in [1.0, 0.1, 0.01]:
#                                                 #     for seed in [981, 982, 983, 984, 985]:
#                                                     for seed in [1103, 1104, 1105]:
#                                                 #     for seed in [73, 74, 75]:
#
#                                                         print(f'@@@ Run gep_public_no_gp SIGMA {sigma} lr {lr} '
#                                                               f'grad_clip {grad_clip} optimizer {optimizer} '
#                                                               f'history_size {history_size} basis_size {basis_size}  %%%')
#
#                                                         sample_prob = float(num_clients) / float(num_client_agg)
#                                                         num_steps = math.ceil(num_epochs * sample_prob)
#                                                         subprocess.run(['poetry', 'run', 'python',
#                                                                         f'trainer_{script_name}_gep_public_no_gp.py',
#                                                                         '--data-name', data_name,
#                                                                         '--classes-per-client', str(classes_per_client),
#                                                                         '--num-steps', str(num_steps),
#                                                                         '--num-clients', str(num_clients),
#                                                                         '--block-size', str(block_size),
#                                                                         '--optimizer', optimizer,
#                                                                         '--lr', str(lr),
#                                                                         '--seed', str(seed),
#                                                                         '--num-client-agg', str(num_client_agg),
#                                                                         '--num-blocks', str(num_blocks),
#                                                                         '--num-private-clients',
#                                                                         str(num_clients - num_public_clients),
#                                                                         '--num-public-clients', str(num_public_clients),
#                                                                         '--noise-multiplier', str(sigma),
#                                                                         '--clip', str(grad_clip),
#                                                                         '--basis-size', str(int(basis_size)),
#                                                                         '--gradients-history-size', str(history_size),
#                                                                         '--csv-name', f'{data_name}_gep_public.csv',
#                                                                         '--eval-after', str(30),
#                                                                         '--eval-every', str(10)
#                                                                         ]
#                                                                        )
#                                                         print(f'<<<<<<<<<< End Run seed {seed} <<<<<<<<<<<<<<<<<<')
#                                                     print(f'<<<<<<<<<< End of clip {grad_clip}')
#                             print(f'<<<<<<<<<< End of sigma {sigma}')
import argparse
import logging
from pathlib import Path
import torch
from fed_trainers.trainers.gep import trainer_cifar10_gep_public_no_gp
from fed_trainers.sweepers.sweep_utils import sweep
from fed_trainers.trainers.utils import set_logger, str2bool


def main():
    parser = argparse.ArgumentParser(
        description="Sweep GEP Public Federated Learning CIFAR10")
    num_users = 500
    ##################################
    #       Network args        #
    ##################################
    # parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10, help="Number of unique labels")
    # parser.add_argument("--num-features", type=int, default=480, help="Number of extracted features (model input size)")

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default='adam',
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
    parser.add_argument("--calibration_split", type=float, default=0.0,
                        help="split ratio of the test set for calibration before testing")
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
        "--data-name", type=str, default="cifar10",
        choices=['cifar10', 'cifar100', 'putEMG'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data_path", type=str, default='./data/', help="dir path for dataset")
    parser.add_argument("--num_clients", type=int, default=num_users, help="total number of clients")
    parser.add_argument("--num_private_clients", type=int, default=num_users - 5, help="number of private clients")
    parser.add_argument("--num_public_clients", type=int, default=5, help="number of public clients")
    parser.add_argument("--classes_per_client", type=int, default=10, help="number of classes each client experience")

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
        "name": f"gep_public_CIFAR10__103to110",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_avg_acc"},
        "parameters": {
            "lr": {"values": [0.1]},
            "global_lr": {"values": [0.999]},
            "seed": {"values": [1103, 1104, 1105]},
            "basis-size": {"values": [args.num_public_clients]},
            "gradients-history-size": {"values": [args.num_public_clients]},
            "num_public_clients": {"values": [args.num_public_clients]},
            "clip": {"values": [5.0, 1.0]},
            "calibration_split": {"values": [0.0, 0.1, 0.2]},
            "noise_multiplier": {"values": [0.0, 2.016, 4.72, 12.79, 25.0]},
            "inner_steps": {"values": [1]},
            "wd": {"values": [0.001]},
            "num_steps": {"values": [2]},
            "num_client_agg": {"values": [5]},
            "depth_power": {"values": [1]}
        },
    }
    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=trainer_cifar10_gep_public_no_gp.train)


if __name__ == '__main__':
    main()
