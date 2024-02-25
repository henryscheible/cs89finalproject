import os
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

# train the model for one ep

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

    processed_train_dataset = []
    for (data, target) in tqdm(train_loader):
            processed_train_dataset += [(data, target)]
    processed_val_dataset = []
    for (data, target) in tqdm(val_loader):
            processed_val_dataset += [(data, target)]
    # for i in range(0, epochs):
    #     print(f"Epoch {i}")
    #     train_dataset_i = []
    #     for (data, target) in tqdm(train_loader):
    #         train_dataset_i += [(data, target)]
    #     processed_train_datasets += [train_dataset_i]

    return processed_train_dataset, processed_val_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)

    parser.add_argument('--datadir', type=str, default="./datasets")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main(args)
