from train import make_model, make_rank_k_model
import torch
import numpy as np
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', type=str, default="model_test.pt")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model_file = torch.load(args.checkpoint_path)
    # nchannels = model_file['args']['nchannels']
    # nunits = model_file['args']['nunits']
    # nclasses = model_file['args']['nclasses']
    # nlayers = model_file['args']['nlayers']
    # rank_constraint = model_file["args"]["rank_constraint"]
    # print(f"nchannels: {nchannels}, nunits: {nunits}, nclasses: {nclasses}, nlayers: {nlayers}, rank_constraint: {rank_constraint}")

    # if model_file["args"]["rank_constraint"] == 0:
    #     model = make_model(nchannels=nchannels, nunits=nunits, nclasses=nclasses, nlayers=nlayers)
    # else:
    #     model = make_rank_k_model(nchannels=nchannels, nunits=nunits, nclasses=nclasses, nlayers=nlayers, k=model_file["args"]["rank_constraint"])

    if "model" in model_file.keys():
        model_file = model_file["model"]
    print(f"Total Parameters: {sum([p.numel() for p in model_file.values()])}")
