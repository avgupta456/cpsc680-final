import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import german

if __name__ == "__main__":
    node_model = torch.load("models/GermanDataset_Node_GCNConv(16,16,1)_Adam_500.pt")
    edge_model = torch.load("models/GermanDataset_Edge_GCNConv(16,16)_Adam_500.pt")

    data = german[0]

    edge_prob_1 = edge_model(data.x, data.edge_index, data.edge_index).sigmoid()
    edge_prob_2 = edge_model.decode(
        node_model.embedding(data.x, data.edge_index), data.edge_index
    ).sigmoid()

    fig, ax = plt.subplots()
    ax.hist(edge_prob_1.detach().numpy(), bins=100)
    ax.hist(edge_prob_2.detach().numpy(), bins=100)
    plt.show()

    edge_prob = edge_prob_1.detach().numpy()

    sens_attrs = data.sens_attrs.flatten().to(int)
    edge_homophily = sens_attrs[data.edge_index[0]] == sens_attrs[data.edge_index[1]]

    # stacked histogram of edge prob 1 for homophily and non-homophily
    fig, ax = plt.subplots()
    ax.hist(
        edge_prob[edge_homophily], bins=100, density=True, alpha=0.5, label="homophily"
    )
    ax.hist(
        edge_prob[~edge_homophily],
        bins=100,
        density=True,
        alpha=0.5,
        label="non-homophily",
    )
    ax.legend()
    plt.show()
