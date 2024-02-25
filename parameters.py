# Regularization experiments

arg_options = {
        "datadir": ["datasets"],
        "nchannels": [3],
        "nclasses": [10],
        "nunits": [256],
        "lr": [0.01],
        "mt": [0.9],
        "batchsize": [256, 1024],
        "epochs": [50],
        "stopcond": [0.01], 
        "l1": [0.0],
        "l2": [0.0, 0.01, 0.1],
        "dropout": [0.0, 0.25, 0.5],
        "nlayers": [3],
        "rank_constraint": [0],
        "data_aug": [0],
        "device": "cuda"
    }



# arg_options = [
#     {
#         "datadir": ["datasets"],
#         "nchannels": [3],
#         "nclasses": [10],
#         "nunits": [256],
#         "lr": [0.01],
#         "mt": [0.9],
#         "batchsize": [256],
#         "epochs": [40],
#         "stopcond": [0.01], 
#         "l1": [0.0],
#         "l2": [0.0],
#         "dropout": [0.0],
#         "nlayers": [3],
#         "rank_constraint": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
#     },
#     {
#         "datadir": ["datasets"],
#         "nchannels": [3],
#         "nclasses": [10],
#         "nunits": [1024],
#         "lr": [0.01],
#         "mt": [0.9],
#         "batchsize": [256],
#         "epochs": [40],
#         "stopcond": [0.01], 
#         "l1": [0.0],
#         "l2": [0.0],
#         "dropout": [0.0],
#         "nlayers": [3],
#         "rank_constraint": [1,2,3,4,5,6,7,8,9,10,11,12,13,20,30,40,50,75,100,150, 200, 220, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
#     },
# ]