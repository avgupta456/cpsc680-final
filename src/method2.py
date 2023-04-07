import torch

from src.datasets import german, german_aware

# method 2 removes modification ratio % of edges in the german dataset that are most sensitive to noise
# re-run the vanilla model and metric evaluation after

if __name__ == "__main__":
    node_model = torch.load("models/GermanDataset_Node_GCNConv(16,1)_Adam_1000.pt")
    node_aware_model = torch.load("models/GermanAwareDataset_Node_GCNConv(16,1)_Adam_1000.pt")

    data = german[0]
    data_aware = german_aware[0]
    unaware_embeddings = node_model.embedding(data.x, data.edge_index)
    aware_embeddings = node_aware_model.embedding(data_aware.x, data_aware.edge_index)

    differences = []

    edge_index = data.edge_index
    for edge in range(data.num_edges):
        unaware_sim = torch.dot(unaware_embeddings[edge_index[0][edge]], unaware_embeddings[edge_index[1][edge]])
        aware_sim = torch.dot(aware_embeddings[edge_index[0][edge]], aware_embeddings[edge_index[1][edge]])
        difference = aware_sim - unaware_sim
        differences.append(difference)

    modification_ratio = .10

    to_remove_indices = torch.topk(torch.tensor(differences), k=int(modification_ratio*data.num_edges)).indices

    print(data.num_edges)

    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    mask[to_remove_indices] = False
    data.edge_index = data.edge_index[:, mask]

    print(data.num_edges)
