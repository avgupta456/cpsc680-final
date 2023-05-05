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


def histogram(data_arr, label_arr, bins=100, density=True, alpha=0.5):
    _, ax = plt.subplots()
    for data, label in zip(data_arr, label_arr):
        ax.hist(
            data,
            bins=bins,
            density=density,
            alpha=alpha,
            label=label,
        )
    ax.legend()
    plt.show()


def make_plots(edge_prob_1, edge_prob_2, edge_homophily):
    histogram(
        [edge_prob_1.detach().numpy(), edge_prob_2.detach().numpy()],
        ["Edge Model", "Node Model"],
    )

    # stacked histogram of edge prob 1 for homophily and non-homophily
    histogram(
        [
            edge_prob_1.detach().numpy()[edge_homophily],
            edge_prob_1.detach().numpy()[~edge_homophily],
        ],
        ["Homophily", "Non-Homophily"],
    )

    # stacked histogram of edge prob 2 for homophily and non-homophily
    histogram(
        [
            edge_prob_2.detach().numpy()[edge_homophily],
            edge_prob_2.detach().numpy()[~edge_homophily],
        ],
        ["Homophily", "Non-Homophily"],
    )


if __name__ == "__main__":
    args = get_args()
    debug, dataset, dataset_name = parse_args(args)

    node_model = torch.load(f"models/{dataset_name}.pt")

    def decode(self, z, edge_label_index):
        # NOTE: In logit space
        x1 = z[edge_label_index[0]]
        x2 = z[edge_label_index[1]]
        return (x1 * x2).sum(dim=1)

    data = dataset[0]
    z = node_model.embedding(data.x, data.edge_index)
    x1 = z[data.edge_index[0]]
    x2 = z[data.edge_index[1]]
    edge_prob = (x1 * x2).sum(dim=1).sigmoid()

    sens_attrs = data.sens_attrs.flatten().to(int)
    edge_homophily = sens_attrs[data.edge_index[0]] == sens_attrs[data.edge_index[1]]

    # Remove homophily edges with low edge probability
    cutoff = edge_prob[edge_homophily].quantile(0.20)
    edge_index = data.edge_index[:, ~(edge_homophily & (edge_prob < cutoff))]

    print(
        f"Removed {data.edge_index.shape[1] - edge_index.shape[1]} edges (out of {data.edge_index.shape[1]}), (cutoff = {cutoff:.2f})"
    )

    dataset_name = args.dataset
    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].edge_index = edge_index

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_edge.pt")
