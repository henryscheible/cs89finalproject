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


if __name__ == "__main__":
    model_path = "models/"
    checkpoint= model_path + 'model_test.pt'
    weight_dict = torch.load(checkpoint)['model']

    k = 128
    convert_weight_to_low_rank(weight_dict, k)


    weights_rank_train = torch.load('models/nlayers=3_k=128.pt',  map_location=torch.device('cpu'))['model state']
    # print(weights_rank_train)
    
    show_all_spec_density(weights_rank_train)
