arg_options = {
    "datadir": ["datasets"],
    "nchannels": [3],
    "nclasses": [10],
    "nunits": range(1, 1000, 10),
    "lr": [0.005],
    "mt": [0.9],
    "batchsize": [256],
    "epochs": [25],
    "stopcond": [0.01],
    "rank_constraint": [0],
    "reg": ["l2"],
}