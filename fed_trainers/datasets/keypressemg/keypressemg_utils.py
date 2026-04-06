from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from fed_trainers.datasets.keypressemg.split import get_split_between_days_dataset
from fed_trainers.datasets.keypressemg.types import Participant, DayT1T2


def load_tests(root: Path, pattern: str) -> np.ndarray:
    tests = []
    for fpath in root.glob(pattern):
        with fpath.open('rb') as f:
            data = np.load(f)
            # logging.debug(f'Loaded {fpath.name} from {root.as_posix()}. shape {data.shape}')
            tests.append(data)
    # logging.debug(f'Loaded {len(tests)} tests from {root}')
    return np.array(tests)



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


def get_user_list():
    return [p.value for p in Participant]

def get_clients(args):
    num_clients = args.num_clients
    num_private_clients = args.num_private_clients
    num_public_clients = args.num_public_clients

    num_dummy_clients = num_clients - (num_private_clients + num_public_clients)

    clients = get_user_list()

    public_clients = clients[:num_public_clients]
    private_clients = clients[num_public_clients:]
    dummy_clients = []

    return public_clients, private_clients, dummy_clients

def get_num_users():
    return len(Participant)

def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = {}, {}, {}

    for p in Participant:
        train_dataset, test_dataset = get_split_between_days_dataset(root = Path(args.data_path),
                                                                                  participant = p,
                                                                                  train_day = DayT1T2.T1,
                                                                                  scale = True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        train_loaders[p.value] = train_loader
        val_loaders[p.value] = eval_loader
        test_loaders[p.value] = test_loader

    return train_loaders, val_loaders, test_loaders

def add_arguments_keypressemg(parser, working_dir: Path):
    parser.add_argument("--num-features", type=int, default=320, help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=20,
                        help="Number of extracted features per channel")
    parser.add_argument("--data_path", type=str,
                            default=(working_dir / 'data/EMG/keypressemg/CleanData/valid_features_long_npy').as_posix(),
    #                         # default=(Path.cwd() / 'data/valid_user_features').as_posix(),
    #                         # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                            help="dir path for dataset")
    return parser