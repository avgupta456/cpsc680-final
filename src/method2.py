import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import german

# method 2 removes modification ratio % of edges in the german dataset that are most sensitive to noise
# re-run the vanilla model and metric evaluation after

if __name__ == "__main__":
    node_model = torch.load("models/GermanDataset_Node_GCNConv(16,16,1)_Adam_500.pt")
    aware_node_model = torch.load("models/AwareGermanDataset__Node_GCNConv(16,16,1)_Adam_500.pt")

    data = german[0]
    unaware_embeddings = node_model.get_embeddings()
    aware_embeddings = aware_node_model.get_embeddings()

    differences = []

    for edge in data.edge_index:
        unaware_sim = torch.dot(unaware_embeddings[edge[0]], unaware_embeddings[edge[1]])
        aware_sim = torch.dot(aware_embeddings[edge[0]], aware_embeddings[edge[1]])
        difference = aware_sim - unaware_sim
        differences.append(difference)

    modification_ratio = .10
    to_remove_indices = torch.topk(torch.tensor(differences), k=int(modification_ratio*data.num_edges)).indices

    data.edge_index = data.edge_index[:, ~torch.tensor(to_remove_indices)]

