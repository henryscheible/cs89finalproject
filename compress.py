import argparse
import torch
import numpy as np
import random


def main(args):
    print(args.checkpoint_path)
    print(args.rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', type=str, default="./models/model_test.pt")
    parser.add_argument('--rank', type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main(args)