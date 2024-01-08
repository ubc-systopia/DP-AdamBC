"""Some data loading module."""
import configlib
from data_utils.jax_dataloader import Normalize, Cast, AddChannelDim

from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CelebA, CIFAR10, FashionMNIST
from torchvision import transforms

import jax
import numpy as np
import os
import glob
import pickle

# Configuration arguments
parser = configlib.add_parser("Dataset config")
parser.add_argument("--data_dir", default='/tmp/data', type=str, metavar='DATA_PATH',
        help="The path where to stop the data.")
parser.add_argument("--dataset", default='MNIST', type=str, metavar='DATASET_NAME',
        help="The dataset to use among: MNIST, CIFAR10.")


class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)

# from https://github.com/deepmind/dm-haiku/issues/18
def restore(array_path: str, n_leaves):
    with open(array_path, "rb") as f:
        flat_state = [np.load(f) for _ in range(n_leaves)]

    return flat_state

class DatasetWithMemory(Dataset):
    def __init__(self, dataset, mem_path, treedef, empty_leaves):
        self.dataset = dataset
        self.mem_path = mem_path
        self.treedef = treedef
        self.n_leaves = len(empty_leaves)
        self.empty_mem = empty_leaves

    def load_mem(self, idx):
        array_path = os.path.join(self.mem_path, f"mem{idx}_data.np")
        if not os.path.isfile(array_path):
            #  array_path = os.path.join(self.mem_path, "memempty_data.np")
            return self.empty_mem
        return restore(array_path, self.n_leaves)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx, self.load_mem(idx))

def get_dataset(conf: configlib.Config):
    "Returns the train, test datasets and respective data loaders"
    if conf.dataset == 'MNIST':
        dataset_fn = get_mnist
    elif conf.dataset == 'FMNIST':
        dataset_fn = get_fmnist
    elif conf.dataset == 'CIFAR10':
        dataset_fn = get_cifar10
    else:
        raise NotImplementedError

    train_set, test_set = _get_dataset(conf, dataset_fn)
    return train_set, test_set

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _get_dataset(conf: configlib.Config, get_dataset_fn):
    ensure_dir(conf.data_dir)
    return get_dataset_fn(conf)

def get_mnist(conf: configlib.Config):
    normalize = [
        #  transforms.RandomCrop(28, padding=4),
        Cast(),
        Normalize([0.5, ], [0.5, ]),
        AddChannelDim(),
    ]
    transform = transforms.Compose(normalize)

    # Load MNIST dataset and use Jax compatible dataloader
    mnist_train = MNIST(conf.data_dir, download=True, transform=transform)
    mnist_test = MNIST(conf.data_dir, train=False, download=True, transform=transform)

    return mnist_train, mnist_test

def get_fmnist(c: configlib.Config):
    normalize = [
        Cast(),
        Normalize([0.5, ], [0.5, ]),
        AddChannelDim(),
    ]
    transform = transforms.Compose(normalize)

    fmnist_train = FashionMNIST(c.data_dir, download=True, transform=transform)
    fmnist_test = FashionMNIST(c.data_dir, train=False, download=True, transform=transform)

    return fmnist_train, fmnist_test

def get_cifar10(conf: configlib.Config):
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        Cast(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_transform = transforms.Compose(augmentations + normalize)
    test_transform = transforms.Compose(normalize)

    cifar10_train = CIFAR10(root=conf.data_dir, train=True, download=True, transform=train_transform)
    cifar10_test = CIFAR10(root=conf.data_dir, train=False, download=True, transform=test_transform)

    return cifar10_train, cifar10_test
