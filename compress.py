import argparse
import torch
from torch import nn
import numpy as np
import random

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
        if 'weight' in key and layers_idx < total_layers-1:
            U, S, V = torch.svd(A)
            U_k = U[:, :k] 
            S_k = S[:k]  
            V_k = V[:, :k]  

            A_approx = torch.mm(U_k, torch.diag(S_k)).mm(V_k.t())
            weights[key] = A_approx
            print(f"Original rank {torch.linalg.matrix_rank(A)} and reconstructed rank {torch.linalg.matrix_rank(A_approx)}")
            layers_idx+=1



def main(args):
    checkpoint_path = args.checkpoint_path
    k = args.rank
    nunits = 1024
    model = nn.Sequential(
        nn.Linear(in_features=10*32*32, out_features=nunits),
        nn.ReLU(),
        nn.Linear(in_features=nunits, out_features=10)
    )
    weights = torch.load(checkpoint_path)
    view_weights(weights)
    convert_weight_to_low_rank(weights, k)



    # model.load_state_dict(torch.load(checkpoint_path))
    # print(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', type=str, default="./models/model_test.pt")
    parser.add_argument('--rank', type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main(args)