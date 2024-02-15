import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt

# train the model for one epoch on the given dataset
def train(model, device, train_loader, criterion, optimizer):
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
def load_cifar10_data(split, datadir):
    # Data Normalization and Augmentation (random cropping and horizontal flipping)
    # The mean and standard deviation of the CIFAR-10 dataset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(p=0.5),
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

# define a fully connected neural network with a single hidden layer
def make_model(nchannels, nunits, nclasses, checkpoint=None):
    # define the model
    model = nn.Sequential(
        nn.Linear(in_features=nchannels*32*32, out_features=nunits),
        nn.ReLU(),
        nn.Linear(in_features=nunits, out_features=nclasses)
    )
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))   

    return model



def main():
    # define the parameters to train your model
    datadir = 'datasets'  # the directory of the dataset
    nchannels = 3
    nclasses = 10
    nunits = 256
    lr = 0.001 
    mt = 0.9
    batchsize = 64
    epochs = 25
    stopcond = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}

    # create an initial model
    model = make_model(nchannels, nunits, nclasses)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mt)

    # loading data
    train_dataset = load_cifar10_data('train', datadir)
    val_dataset = load_cifar10_data('val', datadir)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, **kwargs)

    # training the model
    val_losses = []
    best_acc=0
    checkpoint_path='model1.pt'
    for epoch in range(0, epochs):
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)# Training
        val_acc, val_loss =  validate(model, device, val_loader, criterion)# Validation
        val_losses.append(val_loss)

        print(f'Epoch: {epoch + 1}/{epochs}\t Training loss: {train_loss:.3f}   Training accuracy: {train_acc:.3f}   ',
              f'Validation accuracy: {val_acc:.3f}')

        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # save checkpoint if is a new best
        if is_best:
            torch.save(model.state_dict(), checkpoint_path)
        
        # stop training if the cross-entropy loss is less than the stopping condition
        if train_loss < stopcond:
            break



    # calculate the training error of the learned model

    train_acc, train_loss = validate(model, device, train_loader, criterion)
    val_acc, val_loss = validate(model, device, val_loader, criterion)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    print(f'=================== Summary ===================\n',
          f'Training loss: {train_loss:.3f}   Validation loss {val_loss:.3f}   ',
          f'Training accuracy: {train_acc:.3f}   Validation accuracy: {val_acc:.3f}\n')


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()
