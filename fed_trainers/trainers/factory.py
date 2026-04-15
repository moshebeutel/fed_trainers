import logging
import time
from pathlib import Path
from typing import Any
import torch

from fed_trainers.datasets.dataset import gen_random_loaders
from fed_trainers.trainers.model import CIFAR10_CNN_Tanh, ResNet, get_n_params, FeatureModel, initialize_weights


def get_trainer(args) -> Any:

    if args.dp_method == 'sgd_dp':
        if args.use_gp:
            from fed_trainers.trainers.dp_sgd import trainer_sgd_dp_with_gp as trainer
        else:
            from fed_trainers.trainers.dp_sgd import trainer_sgd_dp_no_gp as trainer
    elif args.dp_method == 'gep_public':
        if args.use_gp:
            from fed_trainers.trainers.gep import trainer_gep_public_with_gp as trainer
        else:
            from fed_trainers.trainers.gep import trainer_gep_public_no_gp as trainer
    elif args.dp_method == 'gep_aux':
        if args.use_gp:
            assert False, 'auxiliary data not supported with gp'
        else:
            from fed_trainers.trainers.gep import trainer_gep_aux_no_gp as trainer
    else:
        raise ValueError(f'Unsupported dp_method: {args.dp_method}')
    return trainer


def get_optimizer(args, network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


def get_clients(args):
    if args.data_name == 'keypressemg':
        from fed_trainers.datasets.keypressemg import keypressemg_utils
        return keypressemg_utils.get_clients(args)

    num_clients = args.num_clients
    num_private_clients = args.num_private_clients
    num_public_clients = args.num_public_clients

    assert num_clients >= (num_private_clients + num_public_clients), \
        f'num clients should be more than sum of all participating clients. Got {num_clients} clients'

    num_dummy_clients = num_clients - (num_private_clients + num_public_clients)

    i = 0
    public_clients = list(range(i, i + num_public_clients))
    i += num_public_clients
    private_clients = list(range(i, i + num_private_clients))
    i += num_private_clients
    dummy_clients = list(range(i, i + num_dummy_clients))
    i += num_dummy_clients

    return public_clients, private_clients, dummy_clients


def get_model(args):
    num_classes = {'cifar10': 10, 'cifar100': 100, 'putEMG': 8, 'mnist': 10, 'femnist': 62, 'keypressemg': 26}[args.data_name]
    in_channels = 1 if args.data_name in ['mnist', 'femnist'] else 3

    if args.data_name in ['cifar10', 'cifar100', 'mnist', 'femnist']:

        assert args.model_name in ['CNNTarget', 'ResNet'], f'Unxpected model name {args.model_name}'

        if args.model_name == 'CNNTarget':
            model = CIFAR10_CNN_Tanh(3)
        else:
            model = ResNet(layers=[args.block_size] * args.num_blocks,
                           num_classes=num_classes, in_channels=in_channels, cls_layer=(not args.use_gp))

    else:
        emg_datasets = ['putEMG', 'keypressemg']
        assert args.data_name in emg_datasets, f'data_name should be one of {emg_datasets}'
        model = FeatureModel(num_features=args.num_features, number_of_classes=args.num_classes, cls_layer=(not args.use_gp),
                             depth_power=args.depth_power)

    initialize_weights(model)

    logger = get_logger(args)
    logger.debug(f'Model 1st layer shape: {next(model.parameters()).shape}')
    logger.info(f'Number Parameters: {get_n_params(model)}')

    return model


def get_dataloaders(args):

    if args.data_name == 'keypressemg':
        from fed_trainers.datasets.keypressemg import keypressemg_utils
        return keypressemg_utils.get_dataloaders(args)
    elif args.data_name == 'putEMG':
        from fed_trainers.datasets import emg_utils
        return emg_utils.get_dataloaders(args)

    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        args.data_name,
        args.data_path,
        args.num_clients,
        args.batch_size,
        args.classes_per_client)

    return train_loaders, val_loaders, test_loaders


def get_logger(args):
    logger = logging.getLogger(args.log_name)
    logger.setLevel(args.log_level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{args.log_name}_{time.asctime()}.log')
    file_handler.setLevel(args.log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
