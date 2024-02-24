from multiprocessing import Pool
from types import SimpleNamespace
import json
from itertools import product
from parameters import arg_options
from train import main
import pandas as pd
import time
import tqdm

base_arg_matrices = [dict(zip(arg_options.keys(), values)) for values in product(*arg_options.values())]
# full_arg_matrices = []
# for arg_matrix in base_arg_matrices:
#    full_arg_matrices += [{**arg_matrix, "rank_constraint": r} for r in range(1, arg_matrix["nunits"], 10)]

arg_matrix = base_arg_matrices
results_list = []

def train_one(args):
    i, args = args
    print(f"Training Model {i}/{len(arg_matrix)}")
    args = SimpleNamespace(**args, checkpoint_path=f"./batched_models/model_{i}.pt")
    start_time = time.time()
    metrics = main(args)
    end_time = time.time()
    return {"model_id": i, **metrics, **vars(args), "training_time": end_time - start_time}


if __name__ == "__main__":
    with Pool(10) as p:
      r = list( \
         tqdm.tqdm(p.imap(train_one, zip(range(len(arg_matrix)), arg_matrix))) \
         )
    results_df = pd.DataFrame.from_records(results_list)
    results_df.to_csv("./results.csv")