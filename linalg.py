import torch
from main import make_model

if __name__ == "__main__":
    model_path = "models/"
    checkpoint= model_path + 'model1.pt'
    nunits = 1024
    model = make_model(3, nunits, 10, checkpoint)
    x = torch.load(checkpoint)
    print(x)
