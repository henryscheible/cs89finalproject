arg_options = {
    "datadir": ["datasets"],
    "nchannels": [3],
    "nclasses": [10],
    "nunits": [500, 1000, 2000],
    "lr": [0.01],
    "mt": [0.9],
    "batchsize": [256],
    "epochs": [40],
    "stopcond": [0.01], 
    "l1": [0.0],
    "l2": [0.0],
    "dropout": [0.0],
    "nlayers": [3],
    "rank_constraint": [0,2,4,6,8,10,20,40,60,80]
}