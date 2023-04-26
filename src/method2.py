import argparse

import matplotlib.pyplot as plt
import torch

from src.argparser import (
    add_dataset_args,
    add_misc_args,
    parse_dataset_args,
    parse_misc_args,
)


def get_args():
    argparser = argparse.ArgumentParser()
    add_misc_args(argparser)
    add_dataset_args(argparser)

    return argparser.parse_args()


def parse_args(args):
    debug = parse_misc_args(args)
    dataset, dataset_name = parse_dataset_args(args)

    return debug, dataset, dataset_name


# method 2 removes modification ratio % of edges in the german dataset that are most sensitive to noise
if __name__ == "__main__":
    # load data and models
    args = get_args()
    debug, dataset, dataset_name = parse_args(args)

    node_model = torch.load(f"models/{dataset_name}.pt")
    data = dataset[0]

    folder_name = dataset_name.split("_")[0]

    if dataset_name.split("_")[-1] == "node":
        new_dataset_name = "_".join(
            dataset_name.split("_")[:-1]
        )  # remove _node from dataset_name
        node_aware_model = torch.load(f"models/{new_dataset_name}_aware_node.pt")
        dataset_aware = torch.load(
            f"data/{folder_name}/processed/{new_dataset_name}_aware_node.pt"
        )
    else:
        node_aware_model = torch.load(f"models/{dataset_name}_aware.pt")
        dataset_aware = torch.load(
            f"data/{folder_name}/processed/{dataset_name}_aware.pt"
        )
    data_aware = dataset_aware[0]

    # get embeddings for each edge from aware and unaware models
    unaware_embeddings = node_model.embedding(data.x, data.edge_index)
    aware_embeddings = node_aware_model.embedding(data_aware.x, data_aware.edge_index)

    # calculate similarity between embeddings for each edge
    differences = []
    edge_index = data.edge_index
    for edge in range(data.num_edges):
        unaware_sim = torch.dot(
            unaware_embeddings[edge_index[0][edge]],
            unaware_embeddings[edge_index[1][edge]],
        )
        aware_sim = torch.dot(
            aware_embeddings[edge_index[0][edge]], aware_embeddings[edge_index[1][edge]]
        )
        difference = aware_sim - unaware_sim
        differences.append(difference)

    # remove modification_ratio % of edges from unaware graph that are most sensitive to noise
    modification_ratio = 0.10
    to_remove_indices = torch.topk(
        torch.tensor(differences), k=int(modification_ratio * data.num_edges)
    ).indices
    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    mask[to_remove_indices] = False
    new_edge_index = data.edge_index[:, mask]

    # save modified dataset
    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].edge_index = new_edge_index
    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_edge_2.pt")

    print(
        f"Removed {data.num_edges - temp[0].num_edges} edges (out of {data.num_edges})"
    )
