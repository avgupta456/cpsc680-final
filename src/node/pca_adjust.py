import argparse

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


if __name__ == "__main__":
    debug, dataset, dataset_name = parse_args(get_args())
    data = dataset[0]

    train_mask = data.train_mask
    sens = data.sens_attrs.flatten()
    x = data.x

    index = 1

    if debug:
        fig, ax = plt.subplots()
        ax.hist(x[sens == 0][:, index], bins=20, alpha=0.5, density=True, label="0")
        ax.hist(x[sens == 1][:, index], bins=20, alpha=0.5, density=True, label="1")
        ax.legend()
        plt.show()

    pca = PCA(n_components=10)
    pca.fit(x[train_mask])

    print(sum(pca.explained_variance_ratio_))

    x = torch.tensor(pca.transform(x)).to(torch.float)

    if debug:
        fig, ax = plt.subplots()
        ax.hist(x[sens == 0][:, index], bins=20, alpha=0.5, density=True, label="0")
        ax.hist(x[sens == 1][:, index], bins=20, alpha=0.5, density=True, label="1")
        ax.legend()
        plt.show()

    folder_name = "pokec" if "pokec" in dataset_name else dataset_name

    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].x = x
    temp[0].num_features = 10

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_modified.pt")
