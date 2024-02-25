import os
from types import SimpleNamespace
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict
from tqdm import tqdm
import pickle


class ProcessedDataLoader():
    def __getitem__(self, i):
        return self.loader[i]



# load and preprocess CIFAR-10 data.
def load_cifar10_data(split, datadir):
    # Data Normalization and Augmentation (random cropping and horizontal flipping)
    # The mean and standard deviation of the CIFAR-10 dataset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=32),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    if split == 'train':
        dataset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    else:
        dataset = datasets.CIFAR10(root=datadir, train=False, download=True, transform=val_transform)
    return dataset

def preprocess_dataset(args):
    # define the parameters to train your model11
    datadir = 'datasets'  # the directory of the dataset

    batchsize = args.batchsize
    epochs = int(args.epochs)

    # loading data
    train_dataset = load_cifar10_data('train', datadir)
    val_dataset = load_cifar10_data('val', datadir)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    p_train_dataset = []
    for (data, target) in tqdm(train_loader):
            p_train_dataset += [(data, target)]
    p_val_dataset = []
    for (data, target) in tqdm(val_loader):
            p_val_dataset += [(data, target)]

    processed_train_dataset = ProcessedDataLoader()
    processed_train_dataset.loader = p_train_dataset
    processed_train_dataset.dataset = train_dataset
    processed_val_dataset = ProcessedDataLoader()
    processed_val_dataset.loader = p_val_dataset
    processed_val_dataset.dataset = val_dataset
    # for i in range(0, epochs):
    #     print(f"Epoch {i}")
    #     train_dataset_i = []
    #     for (data, target) in tqdm(train_loader):
    #         train_dataset_i += [(data, target)]
    #     processed_train_datasets += [train_dataset_i]

    return processed_train_dataset, processed_val_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=25)

    parser.add_argument('--datadir', type=str, default="./datasets")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    processed_train_dataset, processed_val_dataset = preprocess_dataset(args)

    torch.save(processed_train_dataset, "processed_train_dataset.pt")
    torch.save(processed_val_dataset, "processed_val_dataset.pt")


