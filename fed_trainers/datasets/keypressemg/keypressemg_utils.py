from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from fed_trainers.datasets.keypressemg.split import get_split_between_days_dataset, get_same_split_day_datasets
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


def get_user_list():
    # return [p.value for p in Participant]
    return [p.value + d.value for p in Participant for d in DayT1T2]
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
    # return len(Participant)
    return len(Participant) * len(DayT1T2)

def get_dataloaders(args):
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    keep_features, keep_channels = None, None
    if args.num_features_per_channel == 16:
        remove_features = [3,4,5,9]
        keep_features = [i for i in range(20) if i not in remove_features]
    if args.num_features // args.num_features_per_channel == 8:
        keep_channels = [0,1,2,3,4,5,6,7]

    for p in Participant:
        for d in DayT1T2:
            # train_dataset, test_dataset = get_split_between_days_dataset(root=Path(args.data_path), participant=p,
            #                                                              train_day=DayT1T2.T1, scale=True,
            #                                                              features_inds=keep_features,
            #                                                              channels_inds=keep_channels)

            train_dataset, test_dataset = get_same_split_day_datasets(root=Path(args.data_path), participant=p,
                                                                      day=d, scale=False,
                                                                      features_inds=keep_features,
                                                                      channels_inds=keep_channels)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            id = p.value + d.value
            train_loaders[id] = train_loader
            val_loaders[id] = eval_loader
            test_loaders[id] = test_loader

    return train_loaders, val_loaders, test_loaders

def add_arguments_keypressemg(parser, working_dir: Path):
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=256, choices=[256],
                        help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=16,
                        help="Number of extracted features per channel")
    parser.add_argument("--data_path", type=str,
                            # default=(working_dir / 'data/EMG/keypressemg/CleanData/valid_features').as_posix(),
                            default=(working_dir / 'data/EMG/keypressemg/CleanData/valid_features_long_npy').as_posix(),
                            help="dir path for dataset")
    parser.add_argument('--log_data_statistics', type=str, default=True)

    return parser

def add_arguments_keypressemg_as_aux(parser, working_dir: Path):
    parser.add_argument("--aux_data_name", type=str, default="keypressemg")
    parser.add_argument("--aux_num_features", type=int, default=128, help="Number of extracted features (model input size)")
    parser.add_argument("--aux_batch_size", type=int, default=512, help="Batch size for auxiliary data")

    parser.add_argument("--aux_data_path", type=str,
                        # default=(working_dir / 'data/EMG/keypressemg/CleanData/valid_features').as_posix(),
                        default=(working_dir / 'data/EMG/keypressemg/CleanData/valid_features_long_npy').as_posix(),
                        help="dir path for dataset")
    return parser