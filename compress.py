import argparse
import torch
from torch import nn
import numpy as np
from train import make_model, make_rank_k_model, validate, load_cifar10_data
import random
from torch.utils.data import DataLoader

device = 'gpu' if torch.cuda.is_available() else 'cpu'

def view_weights(weights):
    for k, v in weights.items():
        if 'weight' in k:
            print("=============================")
            print(f" Weight layer {k} with shape {v.shape} and has matrix rank of {torch.linalg.matrix_rank(v)}")
                
    print("=============================")


def convert_weight_to_low_rank(weights, k):
    
    layers_idx=0
    total_layers = len(weights)/2
    for key, A in weights.items():
        # Don't adjust the rank of the last layer
        if "weight" in key and layers_idx < total_layers-1:
            print(f"Applying truncated SVD on layer {key}")
            U, S, V = torch.svd(A)
            U_k = U[:, :k] 
            S_k = S[:k]  
            V_k = V[:, :k]  

            A_approx = torch.mm(U_k, torch.diag(S_k)).mm(V_k.t())
            weights[key] = A_approx
            print(f"Original rank {torch.linalg.matrix_rank(A)} and reconstructed rank {torch.linalg.matrix_rank(A_approx)}")
            layers_idx+=1


def eval_model(model, model_name):
    train_dataset = load_cifar10_data('train', './datasets')
    val_dataset = load_cifar10_data('val', './datasets')

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
    checkpoint_path = args.checkpoint_path
    k = args.rank

    model_file = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    weights = model_file['model']

    nchannels = model_file['args']['nchannels']
    nunits = model_file['args']['nunits']
    nclasses = model_file['args']['nclasses']
    nlayers = model_file['args']['nlayers']


    model = make_model(nchannels=nchannels, nunits=nunits, nclasses=nclasses, nlayers=nlayers)
    
    # Do the low-rank approximation of the weights before we evaluate the model 
    convert_weight_to_low_rank(weights, k)
    model.load_state_dict(weights)
    # view_weights(weights)
    

    _, val_acc, _, _ = eval_model(model, f"Model with {k}-Rank Approximation @ Inference")
    print(f"Total Parameters for Truncated Model: {sum([p.numel() for p in weights.values()])}")



    # model_B_file = torch.load("./models/nlayers=3_k=16.pt", map_location=torch.device('cpu'))['model state']
    # print(f"Total Parameters for train rank-constrained Model: {sum([p.numel() for p in model_B_file.values()])}")
    return val_acc
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', type=str, default="model_test.pt")
    parser.add_argument('--rank', type=int, default=16)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main(args)