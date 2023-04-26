import argparse

import torch

from src.argparser import (
    add_dataset_args,
    add_misc_args,
    parse_dataset_args,
    parse_misc_args,
)
from src.datasets import *  # noqa F401 F403


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

    aware_dataset_name = dataset_name + "_aware"
    if "_node" in dataset_name:
        aware_dataset_name = aware_dataset_name.replace("_node", "") + "_node"

    dataset_aware = eval(aware_dataset_name)
    data_aware = dataset_aware[0]

    node_aware_model = torch.load(f"models/{aware_dataset_name}.pt")

    # get embeddings for each edge from aware and unaware models
    unaware_embeddings = node_model.embedding(data.x, data.edge_index)
    aware_embeddings = node_aware_model.embedding(data_aware.x, data_aware.edge_index)

    edge_index = data.edge_index

    unaware_x1 = unaware_embeddings[edge_index[0]]
    unaware_x2 = unaware_embeddings[edge_index[1]]
    unaware_sim = (unaware_x1 * unaware_x2).sum(dim=1)

    aware_x1 = aware_embeddings[edge_index[0]]
    aware_x2 = aware_embeddings[edge_index[1]]
    aware_sim = (aware_x1 * aware_x2).sum(dim=1)

    difference = aware_sim - unaware_sim

    sens_attrs = data.sens_attrs.flatten().to(int)
    edge_homophily = sens_attrs[data.edge_index[0]] == sens_attrs[data.edge_index[1]]

    cutoff = difference[edge_homophily].quantile(0.80)
    new_edge_index = data.edge_index[:, ~(edge_homophily & (difference > cutoff))]

    # save modified dataset
    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].edge_index = new_edge_index
    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_edge.pt")

    print(
        f"Removed {data.num_edges - temp[0].num_edges} edges (out of {data.num_edges})"
    )
