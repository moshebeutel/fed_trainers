import logging
from functools import partial
from pathlib import Path
import wandb
import yaml
from typing import Dict
from fed_trainers.trainers.utils import set_seed

def load_config(file_path: str)-> Dict:
    """
    Loads the configuration from a specified YAML file.

    This function checks the existence, type, and extension of the provided
    file path to ensure it is a valid YAML configuration file. Once validated,
    it reads and parses the YAML content, returning the configuration data.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data as a dictionary.

    Raises:
        AssertionError: If the file does not exist, is not a file, or does not
            have the '.yaml' extension.
    """
    config_file_path = Path(file_path)
    assert config_file_path.exists(), f"config file {config_file_path} does not exist"
    assert config_file_path.is_file(), f"config file {config_file_path} is not a file"
    assert config_file_path.suffix == ".yaml", f"config file {config_file_path} is not a yaml file"
    with open(config_file_path, 'r') as stream:
        sweep_config = yaml.safe_load(stream)
    return sweep_config

def sweep_train(sweep_id, args, train_fn, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        set_seed(config.seed)

        for k, v in config.items():
            if k in args:
                setattr(args, k, v)

        wandb.run.name = '_'.join([f'{k}_{v}' for k, v in config.items()])
        train_fn(args)


def init_sweep(config: Dict) -> int:
    sweep_id = wandb.sweep(sweep=config, project="dec25_sweeps")
    return sweep_id


def start_sweep(sweep_id, f_sweep):
    wandb.agent(sweep_id=sweep_id, function=f_sweep)


def sweep(sweep_config: Dict, args, train_fn):
    logger = logging.getLogger(args.log_name)
    logger.info(f'sweep {sweep_config}')
    sweep_id = init_sweep(sweep_config)
    f_sweep = partial(sweep_train, sweep_id=sweep_id, args=args, train_fn=train_fn)
    # wandb.agent(sweep_id=sweep_id, function=f_sweep)
    start_sweep(sweep_id, f_sweep)
