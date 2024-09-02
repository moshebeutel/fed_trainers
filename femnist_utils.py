from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils import set_logger


def load_data_from_json(json_file: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(json_file, str):
        json_file = Path(json_file)
    assert isinstance(json_file, Path), "json_file must be a path"
    assert json_file.exists(), f"{json_file} does not exist"
    assert json_file.is_file(), f"{json_file} is not a file"
    with open(json_file) as f:
        data = json.load(f)
        # return np.array(data["data"]), np.array(data["labels"])
        return data


def create_user_tensors_from_all_data(args):
    # logger = logging.getLogger(args.log_name)
    logger = set_logger(args)
    data_path = Path(args.data_path)
    assert data_path.exists(), f"{data_path} does not exist"
    assert data_path.is_dir(), f"{data_path} is not a directory"

    logger.info(f"Femnist Loading data from {args.data_path}")

    tensors_path = Path(args.tensors_data_path)
    assert tensors_path.exists(), f'tensors path {tensors_path} does not exist'
    assert tensors_path.is_dir(), f'tensors path {tensors_path} is not a directory'

    users = []
    user_data = {}
    pbar = tqdm(data_path.glob("*.json"))
    for json_file in pbar:
        data_dict = load_data_from_json(json_file)
        users.extend(data_dict["users"])
        logger.debug(f'Loaded users {data_dict["users"]}')
        logger.debug(f'len after {len(users)} len unique {len(list(set(users)))}')
        for user in data_dict['users']:
            user_data = data_dict['user_data'][user]
            user_data_x = torch.tensor(user_data['x'])
            user_data_y = torch.tensor(user_data['y'])
            filename_x = f'user_{user}_x.pt'
            filename_y = f'user_{user}_y.pt'

            torch.save(user_data_x, tensors_path / filename_x)

            torch.save(user_data_y, tensors_path / filename_y)

            pbar.set_postfix({'json file': json_file, 'user': user})

        # current_user_data = {k: np.array(v) for k, v in data_dict["user_data"].items()}
        # user_data = {**user_data, **current_user_data}
        del data_dict


def get_user_list(args):
    tensors_path = Path(args.tensors_data_path)
    assert tensors_path.exists(), f'tensors path {tensors_path} does not exist'
    assert tensors_path.is_dir(), f'tensors path {tensors_path} is not a directory'

    users_file_path = tensors_path / "users.txt"
    assert users_file_path.exists(), f'users path {users_file_path} does not exist'
    with open(users_file_path) as f:
        users = f.read().splitlines()
        return users


def get_dataloaders(args):
    # logger = logging.getLogger(args.log_name)
    logger = set_logger(args)
    tensors_path = Path(args.tensors_data_path)
    assert tensors_path.exists(), f'tensors path {tensors_path} does not exist'
    assert tensors_path.is_dir(), f'tensors path {tensors_path} is not a directory'
    tensor_x_filenames = tensors_path.glob('*x.pt')
    tensor_y_filenames = tensors_path.glob('*y.pt')

    test_split_ratio = args.test_split_ratio
    val_split_ratio = args.val_split_ratio
    train_split_ratio = 1 - test_split_ratio - val_split_ratio
    assert train_split_ratio > 0.5, f'train split is expected to contain the majority of data. Got {train_split_ratio}'

    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for i, fnx in enumerate(tensor_x_filenames):
        fny = Path(fnx.as_posix().replace('_x.pt', '_y.pt'))
        assert fny.exists(), f'y tensor file {fny} for {fnx.as_posix()} does not exit'
        user = fnx.stem.replace('_x', '').replace('user_', '')
        logger.debug(f'user {user}')
        X = torch.load(fnx).reshape(-1, 1, 28, 28)
        logger.debug(f'X shape {X.shape}')
        y = torch.load(fny)
        logger.debug(f'y shape {y.shape}')

        # Define the sizes for train, validation, and test splits
        train_size = int(train_split_ratio * len(X))
        val_size = int(val_split_ratio * len(X))
        test_size = int(test_split_ratio * len(X))

        # Generate shuffled indices
        indices = torch.randperm(len(X))

        # Split the indices into train, val, and test
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Use Subset to create the split datasets
        train_X = X[train_indices]
        train_Y = y[train_indices]

        val_X = X[val_indices]
        val_Y = y[val_indices]

        test_X = X[test_indices]
        test_Y = y[test_indices]

        train_loaders[i] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_X, train_Y),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        val_loaders[i] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_X, val_Y),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        test_loaders[i] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_X, test_Y),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    return train_loaders, val_loaders, test_loaders


if __name__ == "__main__":
    # load_data_from_json("/home/user1/datasets/femnist/all_data/all_data_0.json")
    parser = argparse.ArgumentParser(description="femnist utils")
    parser.add_argument("--data-path", type=str,
                        default=(Path.home() / 'datasets/femnist/all_data').as_posix(),
                        help="dir path for dataset")
    parser.add_argument("--tensors-data-path", type=str,
                        default='data/femnist/user_tensors',
                        help="dir path for dataset")
    parser.add_argument("--log-dir", type=str, default="./log", help="dir path for logger file")
    parser.add_argument("--log-name", type=str, default="femnist_utils", help="dir path for logger file")
    parser.add_argument("--test-split-ratio", type=float, default="0.2", help="The test split len of dataset")
    parser.add_argument("--val-split-ratio", type=float, default="0.1", help="The validation split len of dataset")
    args = parser.parse_args()
    # create_user_tensors_from_all_data(args)
    get_dataloaders(args)
