import logging
import os
import re
from pathlib import Path
from typing import Dict
import torch


def get_user_list():
    return ['03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38', '39', '42', '43', '45',
            '46', '47', '48', '49', '50', '51', '53', '54']

    # return ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    #         '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38', '39', '42', '43', '45',
    #         '46', '47', '48', '49', '50', '51', '53', '54']


def get_num_users():
    return len(get_user_list())


def get_dataloaders(args):
    import pandas as pd
    from biolab_utilities.putemg_utilities import prepare_data, Record, record_filter, data_per_id_and_date

    logger = logging.getLogger(args.log_name)

    # filtered_data_folder = os.path.join(result_folder, 'filtered_data')
    # calculated_features_folder = os.path.join(result_folder, 'calculated_features')
    calculated_features_folder = Path(args.data_path)
    assert calculated_features_folder.exists(), f'{calculated_features_folder} does not exist'
    assert calculated_features_folder.is_dir(), f'{calculated_features_folder} is not a directory'
    assert len(list(
        calculated_features_folder.glob('*.hdf5'))) > 0, f'{calculated_features_folder} does not contain hdf5 files'

    # list all hdf5 files in given input folder
    # all_files = [f.as_posix().replace('_filtered_features', '')
    #              for f in sorted(calculated_features_folder.glob("*_features.hdf5"))]
    all_files = [f.as_posix().replace('_filtered', '')
                 for f in sorted(calculated_features_folder.glob("*_filtered.hdf5"))]

    users_files = []
    users = get_user_list()
    for u in users:
        for f in all_files:
            if f'gestures-{u}' in f:
                users_files.append(f)

    logger.debug(f'{len(users_files)} users found')

    all_files = users_files

    logger.debug(f'Found {len(all_files)} feature files')

    all_feature_records = [Record(os.path.basename(f)) for f in all_files]

    logger.debug(f'Found {len(all_feature_records)} feature records')

    records_filtered_by_subject = record_filter(all_feature_records)

    logger.debug(f'Filtered {len(records_filtered_by_subject)} records')

    splits_all = data_per_id_and_date(records_filtered_by_subject, n_splits=1)

    logger.debug(f'Splits {len(splits_all)}')

    # load feature data to memory
    dfs: Dict[Record, pd.DataFrame] = {}

    for r in records_filtered_by_subject:
        # print("Reading features for input file: ", r)
        filename = os.path.splitext(r.path)[0]
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder, filename + '_filtered.hdf5')))

        # dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder,
        #                                                filename + '_filtered_features.hdf5')))

    logger.debug(f'Found {len(dfs)} dataframes')

    features = ['RMS', 'MAV', 'WL', 'ZC', 'SSC', 'IAV', 'VAR', 'WAMP'] if args.num_features == 8 * 24 else ["IAV",
                                                                                                            "AAC",
                                                                                                            "AR",
                                                                                                            "CC",
                                                                                                            "DASDV",
                                                                                                            "Kurt",
                                                                                                            "LOG",
                                                                                                            "MAV1",
                                                                                                            "MAV2",
                                                                                                            "MAVSLP",
                                                                                                            "MHW",
                                                                                                            "MTW",
                                                                                                            "MYOP",
                                                                                                            'RMS',
                                                                                                            'MAV',
                                                                                                            'WL',
                                                                                                            'ZC',
                                                                                                            'SSC',
                                                                                                            'VAR',
                                                                                                            'WAMP',
                                                                                                            "Skew",
                                                                                                            "SSI",
                                                                                                            "TM",
                                                                                                            "V",
                                                                                                            "MNF",
                                                                                                            "MDF",
                                                                                                            "PKF",
                                                                                                            "MNP",
                                                                                                            "TTP",
                                                                                                            "FR",
                                                                                                            "VCF",
                                                                                                            "PSR",
                                                                                                            "SNR",
                                                                                                            "DPR",
                                                                                                            "OHM",
                                                                                                            "MAX",
                                                                                                            "SMR"]

    assert (len(features) * 24) == args.num_features, f'Expected num features: {len(features) * 24}. Do not match args'

    # defines gestures to be used in shallow learn
    gestures = {
        0: "Idle",
        1: "Fist",
        2: "Flexion",
        3: "Extension",
        4: "Pinch index",
        5: "Pinch middle",
        6: "Pinch ring",
        7: "Pinch small"
    }
    channel_range = {
        "24chn": {"begin": 1, "end": 24},
        # "8chn_1band": {"begin": 1, "end": 8},
        "8chn_2band": {"begin": 9, "end": 16},
        # "8chn_3band": {"begin": 17, "end": 24}
    }
    ch_range = channel_range['24chn']

    num_clients = len(splits_all.values())
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for id in range(num_clients // 2):
        train_x_s, test_x_s = [], []
        train_y_s, test_y_s = [], []
        for client_id in [2 * id, 2 * id + 1]:
            # iterate over each internal data
            for i_s, subject_data in enumerate(list(splits_all.values())[client_id]):
                # get data of client
                # prepare training and testing set based on combination of k-fold split, feature set and gesture set
                # this is also where gesture transitions are deleted from training and test set
                # only active part of gesture performance remains
                data = prepare_data(dfs, subject_data, features, list(gestures.keys()))

                logger.debug(f'Processing subject {i_s}:  {subject_data}')
                logger.debug(f'For client: {client_id}')

                # list columns containing only feature data
                regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
                cols = list(filter(regex.search, list(data["train"].columns.values)))

                logger.debug(f'Found {len(cols)} columns')

                # strip columns to include only selected channels, eg. only one band
                cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]

                logger.debug(f'Found {len(cols)} columns after strip')

                # extract limited training x and y, only with chosen channel configuration
                train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
                train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())
                train_y[train_y > 5] -= 2

                logger.debug(f'Train data shape: {train_x.shape}')

                # # extract limited testing x and y, only with chosen channel configuration
                test_x = torch.tensor(data["test"][cols].to_numpy(), dtype=torch.float32)
                test_y_true = torch.LongTensor(data["test"]["output_0"].to_numpy())
                test_y_true[test_y_true > 5] -= 2

                logger.debug(f'Test data shape: {test_x.shape}')

                train_x_s.append(train_x)
                test_x_s.append(test_x)
                train_y_s.append(train_y)
                test_y_s.append(test_y_true)

                logger.debug(f'Train data list length: {len(train_x_s)}')
                logger.debug(f'Test data list length: {len(test_x_s)}')

        train_loaders[id] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x_s[0], train_y_s[0]),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        val_loaders[id] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x_s[1], train_y_s[1]),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        test_loaders[id] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_x, test_y_true),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        logger.debug(f'Train loaders {list(train_loaders.keys())}')

    return train_loaders, val_loaders, test_loaders


def get_optimizer(args, network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)
