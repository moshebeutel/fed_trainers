import random
from collections import defaultdict
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from typing import List, Sequence, Optional

def federated_split(
    y: Sequence[int],
    n_clients: int,
    alpha: float = 1.0,
    min_per_client: int = 0,
    seed: Optional[int] = 42,
    strict_iid_at_one: bool = True,
) -> List[List[int]]:
    """
    Split a labeled dataset into n_clients partitions for federated learning.

    Parameters
    ----------
    y : Sequence[int]
        Labels for each sample (len(y) = number of samples). Can be list/np.ndarray/torch.Tensor/pd.Series.
    n_clients : int
        Number of clients.
    alpha : float, default 1.0
        User-friendly heterogeneity control where alpha=1.0 => IID.
        For alpha in [0, 1):
           We map alpha to a Dirichlet concentration c = alpha / (1 - alpha).
           Smaller alpha => more non-IID; alpha -> 0 => highly skewed.
        For alpha == 1.0:
           We perform a stratified IID split (equal class distribution per client).
    min_per_client : int, default 0
        If >0, ensures each client gets at least this many samples (soft constraint via redraws).
    seed : int or None
        RNG seed for reproducibility. If None, randomness is not seeded.
    strict_iid_at_one : bool
        If True and alpha==1, enforce exact stratified IID (recommended).

    Returns
    -------
    List[List[int]]
        A list of index lists: indices_per_client[i] are the sample indices assigned to client i.

    Notes
    -----
    - Dirichlet-based non-IID split: For each class, we draw proportions ~ Dirichlet(c, ..., c)
      and split that class' indices according to those proportions.
    - When alpha==1 and strict_iid_at_one is True, each class is evenly and randomly distributed
      across clients (perfectly IID up to remainder effects).
    - If min_per_client>0, we redraw the entire allocation until all clients meet the minimum
      or we hit a reasonable retry limit.
    """

    # Convert labels to numpy array
    y = np.asarray(y)
    n_samples = len(y)
    assert n_clients >= 1, "n_clients must be >= 1"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1], with alpha=1 meaning IID"

    rng = np.random.default_rng(seed)

    classes, y_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # --- Helper: Stratified IID split (alpha == 1 case) ---
    def stratified_iid_split() -> List[List[int]]:
        indices_per_client = [[] for _ in range(n_clients)]
        # For each class, shuffle and split evenly across clients
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            # Split approximately evenly across clients
            splits = np.array_split(cls_idx, n_clients)
            for i, part in enumerate(splits):
                indices_per_client[i].extend(part.tolist())
        # Shuffle within client for randomness
        for i in range(n_clients):
            rng.shuffle(indices_per_client[i])
        return indices_per_client

    # Fast path for exact IID
    if strict_iid_at_one and np.isclose(alpha, 1.0):
        return stratified_iid_split()

    # --- Dirichlet-based non-IID split for alpha in [0,1) ---
    # Map alpha in [0,1) to a Dirichlet concentration c in [0, ∞), with alpha=1 -> c=∞ (handled above)
    # c = alpha / (1 - alpha); clamp to a small positive minimum for numerical stability
    c = max(alpha / max(1.0 - alpha, 1e-12), 1e-6)

    # Try multiple draws to satisfy min_per_client if requested
    max_redraws = 200
    for _ in range(max_redraws):
        indices_per_client = [[] for _ in range(n_clients)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            # Draw class proportions across clients
            proportions = rng.dirichlet(alpha=[c] * n_clients)
            # Translate proportions into integer split sizes
            counts = np.floor(proportions * len(cls_idx)).astype(int)

            # Adjust counts to sum exactly to len(cls_idx) (due to flooring)
            deficit = len(cls_idx) - counts.sum()
            if deficit > 0:
                # Distribute leftover items to the largest fractional parts
                # (use the remainder ranking)
                remainders = proportions * len(cls_idx) - counts
                assign_order = np.argsort(-remainders)  # descending
                for j in assign_order[:deficit]:
                    counts[j] += 1

            # Now split class indices accordingly
            assert counts.sum() == len(cls_idx)
            if len(cls_idx) == 0:
                continue
            splits = []
            start = 0
            for cnt in counts:
                end = start + int(cnt)
                splits.append(cls_idx[start:end])
                start = end
            for i, part in enumerate(splits):
                if len(part) > 0:
                    indices_per_client[i].extend(part.tolist())

        # Optionally enforce min_per_client via redraws
        if min_per_client > 0:
            sizes = [len(lst) for lst in indices_per_client]
            if min(sizes) >= min_per_client:
                # Shuffle within each client for randomness
                for i in range(n_clients):
                    rng.shuffle(indices_per_client[i])
                return indices_per_client
        else:
            for i in range(n_clients):
                rng.shuffle(indices_per_client[i])
            return indices_per_client

    # If we land here, we failed min_per_client constraint within max_redraws: fall back to IID
    return stratified_iid_split()


# -------------------------------
# Example usage with PyTorch data:
# -------------------------------
if __name__ == "__main__":
    # Example with synthetic labels:
    # Suppose 10,000 samples, 10 classes, uniform labels
    n = 10_000
    n_clients = 5
    labels = np.repeat(np.arange(10), n // 10)

    # alpha = 1.0 -> IID
    idxs_iid = federated_split(labels, n_clients=n_clients, alpha=1.0, seed=0)
    sizes_iid = [len(i) for i in idxs_iid]
    print("IID sizes:", sizes_iid)

    # alpha = 0.3 -> noticeable non-IID
    idxs_noniid = federated_split(labels, n_clients=n_clients, alpha=0.3, min_per_client=500, seed=0)
    sizes_noniid = [len(i) for i in idxs_noniid]
    print("non-IID sizes:", sizes_noniid)

    # If you have a PyTorch dataset `dataset`, you can do:
    # from torch.utils.data import Subset, DataLoader
    # loaders = [
    #     DataLoader(Subset(dataset, idxs), batch_size=64, shuffle=True, num_workers=2)
    #     for idxs in idxs_noniid
    # ]


def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
    elif data_name == 'mnist':
        normalization = transforms.Normalize((0.1307,), (0.3081,))
        data_obj = MNIST

    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)

    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes + 1
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=True)
    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        # create dataloaders from subsets
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders