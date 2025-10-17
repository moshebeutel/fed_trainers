from pathlib import Path

from keypressemg.common.types_defined import Participant, DayT1T2
from keypressemg.datasets.split_between_days_dataset import get_split_between_days_dataset
from torch.utils.data import DataLoader


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