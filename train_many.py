from types import SimpleNamespace
import json
from itertools import product
from parameters import arg_options
from train import main
import pandas as pd

arg_matrix = [dict(zip(arg_options.keys(), values)) for values in product(*arg_options.values())]

results_list = []

for i, args in enumerate(arg_matrix):
    print(f"Training Model {i}")
    args = SimpleNamespace(**args, checkpoint_path="./batched_models/model_{i}.pt")
    metrics = main(args)
    results_list.append({"model_id": i, **metrics, **args})

results_df = pd.DataFrame.from_records(results_list)
results_df.to_csv("./results.csv")

