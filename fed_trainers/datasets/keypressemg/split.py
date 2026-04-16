import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset

from fed_trainers.datasets.keypressemg.types import Participant, DayT1T2


def load_X_y(root: Path, participant: Participant, experiment_day: DayT1T2) -> Tuple[np.ndarray, np.ndarray]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    arrays = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_X.npy'):
        with open(filename.as_posix(), 'rb') as f:
            arr = np.load(f)
            arrays.append(arr)
    X = np.concatenate(arrays, axis=0)

    labels = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_y.npy'):
        with open(filename.as_posix(), 'rb') as f:
            l = np.load(f)
            labels.append(l)
    y = np.concatenate(labels, axis=0)

    assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'
    return X, y

def get_same_split_day_arrays(root: Path,
                              participant: Participant,
                              day: DayT1T2,
                              split_ratio: float = 0.8,
                              shuffle: bool = True, scale: bool = True):
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X, y = load_X_y(root, participant, day)

    num_recordings = len(X)
    assert num_recordings == len(y), f'Expected {num_recordings} labels, but got {len(y)}'

    if shuffle:
        shuffled_indices = torch.randperm(num_recordings)
        X, y = X[shuffled_indices], y[shuffled_indices]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    split_index = int(split_ratio * num_recordings)

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


def get_same_split_day_datasets(root: Path,
                                participant: Participant,
                                day: DayT1T2,
                                split_ratio: float = 0.8,
                                shuffle: bool = True,
                                scale: bool = True,
                                features_inds: List[int] = None,
                                channels_inds: List[int] = None
    ) -> Tuple[Dataset, Dataset]:
    """
    Splits the recordings for a given participant and day into train and test Datasets.

    Args:
        root (Path): The root directory where the data is stored.
        participant (Participant): The participant for whom the data is being loaded.
        day (DayT1T2): The specific day of testing (one or two).
        split_ratio (float, optional): The ratio of the data to be used for training.
            Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
            Defaults to True.
        scale (bool, optional): Whether to scale the data before splitting.
            Defaults to True.
        features_inds (List[int]): Indices of features to retain from the dataset. If None, all features are used.
        channels_inds (List[int]): Indices of channels to retain from the dataset. If None, all c


    Returns:
        tuple[Dataset, Dataset]: A tuple containing the training and testing datasets
            as `TensorDataset` objects.

    Raises:
        AssertionError: If the root path does not exist or is not a directory.
        AssertionError: If the number of loaded recordings does not match the number of loaded labels.
    """

    X_train, y_train, X_test, y_test = get_same_split_day_arrays(root, participant, day, split_ratio, shuffle, scale)
    if features_inds is not None:
        X_train = X_train.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16 * len(features_inds))
        X_test = X_test.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16 * len(features_inds))
    if channels_inds is not None:
        X_train = X_train.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16 * len(channels_inds))
        X_test = X_test.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16 * len(channels_inds))

    return (TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            TensorDataset(torch.from_numpy(X_test).float(),
                          torch.from_numpy(y_test).long()))




def get_split_between_days_arrays(root: Path,
                                  participant: Participant,
                                  train_day: DayT1T2,
                                  scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X_train, y_train = load_X_y(root, participant, train_day)
    assert len(X_train) == len(y_train), (f'Expected same number of labels and signal windows.'
                                          f' Got {len(X_train)} windows != {len(y_train)} labels')
    X_test, y_test = load_X_y(root, participant, train_day.other_day())
    assert len(X_test) == len(y_test), (f'Expected same number of labels and signal windows.'
                                        f'Got {len(X_test)} windows != {len(y_test)} labels')

    num_samples_train = X_train.shape[0]
    assert num_samples_train == y_train.shape[0], (f'Expected same number of data samples and targets.'
                                                   f' Got {num_samples_train} data samples'
                                                   f' and {y_train.shape[0]} targets')

    num_samples_test = X_test.shape[0]
    assert num_samples_test == y_test.shape[0], (f'Expected same number of data samples and targets.'
                                                 f' Got {num_samples_test} data samples'
                                                 f' and {y_test.shape[0]} targets')
    if scale:
        data = np.concatenate([X_train, X_test], axis=0)
        assert data.shape[0] == num_samples_train + num_samples_test

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        X_train = data_scaled[:num_samples_train]
        X_test = data_scaled[num_samples_train:]

    return X_train, y_train, X_test, y_test


def get_split_between_days_dataset(root: Path, participant: Participant, train_day: DayT1T2, scale: bool = True,
                                   features_inds: List[int] = None,
                                   channels_inds: List[int] = None):
    """
    Splits and preprocesses a dataset into training and test sets based on participant and day information.

    The function loads the dataset for a given participant and training day, then splits it by day, ensuring that
    the training and test datasets correspond to different days. The data can optionally be scaled and filtered
    for specific feature or channel indices.

    Args:
        root (Path): Root directory containing the dataset.
        participant (Participant): Participant identifier for the dataset.
        train_day (DayT1T2): Training day identifier. This determines the split based on the other day.
        scale (bool): Whether to scale the data using a predefined scaling approach. Default is True.
        features_inds (List[int]): Indices of features to retain from the dataset. If None, all features are used.
        channels_inds (List[int]): Indices of channels to retain from the dataset. If None, all channels are used.

    Returns:
        Tuple[TensorDataset, TensorDataset]: A tuple containing the training dataset and test dataset.
    """
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X_train, y_train = load_X_y(root, participant, train_day)
    assert len(X_train) == len(y_train), (f'Expected same number of labels and signal windows.'
                                          f' Got {len(X_train)} windows != {len(y_train)} labels')
    X_test, y_test = load_X_y(root, participant, train_day.other_day())
    assert len(X_test) == len(y_test), (f'Expected same number of labels and signal windows.'
                                        f'Got {len(X_test)} windows != {len(y_test)} labels')

    X_train, y_train, X_test, y_test = get_split_between_days_arrays(root, participant, train_day, scale)
    if features_inds is not None:
        X_train = X_train.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16*len(features_inds))
        X_test = X_test.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16*len(features_inds))
    if channels_inds is not None:
        X_train = X_train.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16*len(channels_inds))
        X_test = X_test.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16*len(channels_inds))


    return (TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            TensorDataset(torch.from_numpy(X_test).float(),
                          torch.from_numpy(y_test).long()))

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATA_ROOT = Path().home() / 'data/datasets/EMG/keypressemg/CleanData'
    VALID_FEATURES_ROOT = DATA_ROOT / 'valid_features'
    VALID_WINDOWS_ROOT = DATA_ROOT / 'valid_windows'

    logging.basicConfig(level=logging.INFO)
    logging.info("Split between days")
    train_ds, test_ds = get_split_between_days_dataset(root=VALID_WINDOWS_ROOT, participant=Participant.P1,
                                                       train_day=DayT1T2.T1, scale=True)
    logging.info(f'dataset contains {ds.__len__()} windows')
    trainloader = DataLoader(train_ds, batch_size=8, shuffle=True)
    for batch_windows, batch_labels in trainloader:
        logging.info(f'batch_windows shape: {batch_windows.shape}, '
                     f'labels shape: {len(batch_labels)}')
        logging.info(f'batch_labels: {batch_labels}')

    logging.info("Split same day")
    length = 0
    for p in Participant:
        for t in DayT1T2:
            train_ds, test_ds = get_same_split_day_datasets(root=VALID_FEATURES_ROOT, participant=p, day=t, scale=True)

            logging.info(f'train ds len {train_ds.__len__()}')
            logging.info(f'test ds len {test_ds.__len__()}')
            d, l = next(iter(train_ds))
            logging.info(f'train data shape {d.shape} l {l}')
            d, l = next(iter(test_ds))
            logging.info(f'test data shape {d.shape} l {l}')

