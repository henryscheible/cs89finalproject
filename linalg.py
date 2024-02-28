import argparse
import torch
import matplotlib.pyplot as plt
from compress import convert_weight_to_low_rank

def show_spec_density(weight_layer_name, weight_matrix, fig, axs, i=0,):
    # Compute the singular value decomposition

    U, S, V = torch.svd(weight_matrix)
    # Compute the singular values
    singular_values = S.cpu().detach().numpy()
    # Plot the singular values

    axs[i].hist(singular_values)
    axs[i].set_title(f'ESD for {weight_layer_name}', fontsize=9)
    axs[i].set_xlabel('Singular Value')
    axs[i].set_ylabel('Freq.')
    # plt.show()

def show_all_spec_density(weight_dict):
    fig, axs = plt.subplots(1, int(len(weight_dict)/2), figsize=(20, 8))
    weight_idx=0
    for k, v in weight_dict.items():
        if 'weight' not in k:
            continue
        show_spec_density(k, v, fig, axs, weight_idx)
        weight_idx+=1
    plt.subplots_adjust(wspace=.5)
    plt.show()


def show_spec_overlayed(regularized, weight):
    # Compute the singular value decomposition

    

    U, S, V = torch.svd(regularized)
    singular_values1 = S.cpu().detach().numpy()

    print(singular_values1)

    U, S, V = torch.svd(weight)
    singular_values2 = S.cpu().detach().numpy()

    plt.hist(singular_values1, bins=50, alpha=0.5, label='Regularized')
    plt.hist(singular_values2, bins=50, alpha=0.5, label='Original')
    plt.title('Singular Value Distribution')
    plt.xlabel('Singular Value')
    plt.ylabel('Freq.')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default="model_1.pt")
    args = parser.parse_args()

    model_path = "models/"
    checkpoint= model_path + args.checkpoint_path
    print(checkpoint)
    weight_dict = torch.load(checkpoint,  map_location=torch.device('cpu'))['model']

    k = 128
    convert_weight_to_low_rank(weight_dict, k)


    weights_rank_train = torch.load('models/nlayers=3_k=128.pt',  map_location=torch.device('cpu'))['model state']
    # print(weights_rank_train)
    
    weights_model_test = torch.load('models/model_test.pt',  map_location=torch.device('cpu'))['model']


    # show_all_spec_density(weight_dict)

     
    show_spec_overlayed(weight_dict['layer 0.weight'], weights_model_test['fc1.weight'])
