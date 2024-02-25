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
from preprocess_dataset import ProcessedDataLoader

import subprocess

# train the model for one epoch on the given dataset
def train(model, device, train_loader, criterion, optimizer, l1_lambda):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()
    for data, target in train_loader:
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()

        if l1_lambda >0:
            l1_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
            loss = loss + l1_lambda * l1_reg
        
        loss.backward()
        optimizer.step()

    train_accuracy = sum_correct / len(train_loader.dataset)
    train_loss = sum_loss / len(train_loader.dataset)
    return train_accuracy, train_loss

# evaluate the model on the given set
def validate(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()


    val_accuracy = sum_correct / len(val_loader.dataset)
    val_loss = sum_loss / len(val_loader.dataset)
    return val_accuracy, val_loss


# load and preprocess CIFAR-10 data.
def load_cifar10_data(split, datadir, data_aug=0):
    # Data Normalization and Augmentation (random cropping and horizontal flipping)
    # The mean and standard deviation of the CIFAR-10 dataset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # if we are not doing data-augmentation, only apply normalization to training data
    if not bool(data_aug): 
        print(f"Data augmentation on the {split} set: FALSE")
        train_transform = val_transform
    else:
        print(f"Data augmentation on the {split} set: TRUE")

    if split == 'train':
        dataset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    else:
        dataset = datasets.CIFAR10(root=datadir, train=False, download=True, transform=val_transform)
    return dataset

# define a fully connected neural network with a single hidden layer
def make_model(nchannels, nunits, nclasses, checkpoint=None, nlayers=1):
    layers = OrderedDict() # container to store layers 

    # first layer
    layers['fc1'] = nn.Linear(in_features=nchannels*32*32, out_features=nunits)
    layers["fc1_non_lin"] = nn.ReLU()

    # middle layers 
    for layer_num in range(nlayers-1):
        layers[f'fc{layer_num + 2}'] = nn.Linear(in_features=nunits, out_features=nunits)
        layers[f'fc{layer_num + 2}_non_lin'] = nn.ReLU()

    # final layer; projects onto number of classes
    layers["classification"] = nn.Linear(in_features=nunits, out_features=nclasses)

    # 2) Define model
    model = nn.Sequential(layers)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))   
    return model


# define NN where we enforce the rank of each weight matrix to be k
def make_rank_k_model(nchannels, nunits, nclasses, nlayers, k, checkpoint=None):
    """
    Function to make a feed forward network where each of the weight matricies have rank <= k. 
    
    Each "normal" layer [a map from dimention  d -> d'] in the feed forward network will be composed into 2 layers;
    the first layer projects the d dimentional input into k dimentional space and the second pushes the k dim vector back up into d'.
    
    The resulting linear map is thus at most rank k.

    inputs:
    - nchannels=3 for CIFAR-10
    - nunits: number of hidden units in each layer 
    - n_layers: number of layers in network
    - k: upper bound on the rank of all parameter matricies in the network

    outputs:
    - model: type torch.nn.Module
    """

    # 1) Construct layers of the network 

    layers = OrderedDict() # container to store layers 

    # first layer
    layers['fc1_down'] = nn.Linear(in_features=nchannels*32*32, out_features=k)
    layers['fc1_up'] = nn.Linear(in_features=k, out_features=nunits)
    layers["fc1_non_lin"] = nn.ReLU()

    # middle layers 
    for layer_num in range(nlayers-1):
        layers[f'fc{layer_num + 2}_down'] = nn.Linear(in_features=nunits, out_features=k)
        layers[f'fc{layer_num + 2}_up'] = nn.Linear(in_features=k, out_features=nunits)
        layers[f'fc{layer_num + 2}_non_lin'] = nn.ReLU()

    # final layer; projects onto number of classes
    layers["classification"] = nn.Linear(in_features=nunits, out_features=nclasses)

    # 2) Define model
    model = nn.Sequential(layers)

    # 3) Load model if checkpoint given
    if checkpoint: model.load_state_dict(torch.load(checkpoint))  
    
    return model


def main(args):
    # define the parameters to train your model11
    datadir = 'datasets'  # the directory of the dataset
    nchannels = args.nchannels
    nclasses = args.nclasses

    data_aug = args.data_aug
    nunits = args.nunits
    nlayers = args.nlayers
    lr = args.lr
    mt = args.mt
    batchsize = args.batchsize
    epochs = args.epochs
    stopcond = args.stopcond
    rank_constraint = args.rank_constraint
    l1_lambda = args.l1
    l2_lambda = args.l2
    dropout_p = args.dropout

    weight_decay = l2_lambda
    # print(f"Running L2 (weight) decay of {weight_decay}")
    # print(f"Running L1 decay of {l1_lambda}")
    # print(f"Running dropout where prob of dropout is {dropout_p}")
    # print(f"")

    device = args.device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}

    # create an initial model; do a check to see if we are ensuring the rank of all weight matricies <= k
    if args.rank_constraint > 0:
        # print(f"Constructing model with rank {rank_constraint} constraint")
        model = make_rank_k_model(nchannels, nunits, nclasses, nlayers=nlayers, k=rank_constraint)
    else:
        # print(f"Constructing normal model with no rank constrant")
        model = make_model(nchannels, nunits, nclasses, nlayers=nlayers)
    

    if dropout_p>0:
        dropout_model = nn.Sequential()
        for i, layer in enumerate(model):
            dropout_model.add_module(f"layer {i}", layer)
            if isinstance(layer, nn.ReLU):
                dropout_model.add_module(f"dropout after layer {i}", nn.Dropout(p=dropout_p))
        model = dropout_model

    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mt, weight_decay=weight_decay)

    # loading data
    if not args.train_dataset_path:
        train_dataset = load_cifar10_data('train', datadir, data_aug)
        val_dataset = load_cifar10_data('val', datadir)

        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, **kwargs)
    else:
        val_loader = torch.load(args.val_dataset_path)
        train_loader = torch.load(args.train_dataset_path)

    # training the model
    val_losses = []
    best_acc=0
    checkpoint_path=args.checkpoint_path
    for epoch in tqdm(range(0, epochs)):
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer, l1_lambda)# Training
        val_acc, val_loss =  validate(model, device, val_loader, criterion)# Validation
        val_losses.append(val_loss)

        # print(f'Epoch: {epoch + 1}/{epochs}\t Training loss: {train_loss:.3f}   Training accuracy: {train_acc:.3f}   ',
        #       f'Validation accuracy: {val_acc:.3f}')

        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # save checkpoint if is a new best
        if is_best:
            torch.save({"model": model.state_dict(), "args": vars(args)}, checkpoint_path)
        
        # stop training if the cross-entropy loss is less than the stopping condition
        if train_loss < stopcond:
            break

    # calculate the training error of the learned model
    best_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(best_state_dict["model"])

    train_acc, train_loss = validate(model, device, train_loader, criterion)
    val_acc, val_loss = validate(model, device, val_loader, criterion)

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()

    print(f'=================== Summary ===================\n',
          f'Training loss: {train_loss:.3f}   Validation loss {val_loss:.3f}   ',
          f'Training accuracy: {train_acc:.3f}   Validation accuracy: {val_acc:.3f}\n')
    
    metrics = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss
    }

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    defaultdevice = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser.add_argument('--data-aug', type=int, default=1)
    parser.add_argument('--device', type=str, default=defaultdevice)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--datadir', type=str, default="./datasets")
    parser.add_argument('--nchannels', type=int, default=3)
    parser.add_argument('--nclasses', type=int, default=10)
    parser.add_argument('--nunits', type=int, default=256)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--stopcond', type=float, default=0.01)
    parser.add_argument('--rank_constraint', type=int, default=0)
    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--train-dataset-path', type=str, default=None)
    parser.add_argument('--val-dataset-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default="./models/model_test.pt")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main(args)
