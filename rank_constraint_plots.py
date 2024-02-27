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
import json
from preprocess_dataset import ProcessedDataLoader
import argparse
import subprocess
from train import load_cifar10_data, validate, make_rank_k_model


def eval_model(model, model_name, device, data_aug=1):
    train_dataset = load_cifar10_data('train', './datasets',data_aug)
    val_dataset = load_cifar10_data('val', './datasets',data_aug)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    train_acc, train_loss = validate(model, device, train_loader, criterion)
    val_acc, val_loss = validate(model, device, val_loader, criterion)

    print(f'=================== Summary for model {model_name}===================\n',
          f'Training loss: {train_loss:.3f}   Validation loss {val_loss:.3f}   ',
          f'Training accuracy: {train_acc:.3f}   Validation accuracy: {val_acc:.3f}\n')

    return train_acc, val_acc, train_loss, val_loss


def main(args):


    device = args.device
    root_dir = args.rootdir
    data_aug = args.data_aug


    val_accs = []
    rank_constraints = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                # Construct the full path to the file
                file_path = os.path.join(dirpath, filename)
                model_obj = torch.load(file_path, map_location=torch.device(device))

                # for the hps that were saved in a dumb way

                # need to check how we saved the hps.
                if "model state" in model_obj.keys():
                    weights = model_obj['model state']
                    congif = model_obj['hps']

                    nchannels = congif.nchannels
                    nunits = congif.nunits
                    nclasses = congif.nclasses
                    nlayers = congif.nlayers
                    rank_constraint = congif.rank_constraint
               
                else:
                    weights = model_obj['model']
                    nchannels = model_obj['args']['nchannels'] 
                    nunits = model_obj['args']['nunits'] 
                    nclasses = model_obj['args']['nclasses']
                    nlayers = model_obj['args']['nlayers'] 
                    rank_constraint = model_obj['args']['rank_constraint'] 

                model = make_rank_k_model(nchannels=nchannels, nunits=nunits, nclasses=nclasses, nlayers=nlayers, k=rank_constraint)
                model.load_state_dict(weights)

                train_acc, val_acc, train_loss, val_loss = eval_model(model, f"Model with each weight layer having rank at most {rank_constraint} during training", device, data_aug)
                
                print(f"Model with rank constraint= {rank_constraint} got val acc of {val_acc}")
                val_accs.append(val_acc)
                rank_constraints.append(rank_constraint)
    
    rank_constraints = np.array(rank_constraints)
    val_accs = np.array(val_accs)

    print(rank_constraints.shape)
    print(val_accs.shape)

    output = {
        "rank_constraints": rank_constraints.tolist(),
        "val_accs": val_accs.tolist(),
        "nunits": nunits,
        "data_aug": data_aug
    }

    with open(f'./figures/Rank_Reduced_Training_nunits={nunits}_dataaug={bool(data_aug)}.json', "w") as f:
        json.dump(output, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=1)
    parser.add_argument('--rootdir', type=str)
    parser.add_argument('--data_aug', type=int)
    args = parser.parse_args()

    main(args)