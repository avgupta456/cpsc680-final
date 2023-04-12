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


if __name__ == "__main__":
    debug, dataset, dataset_name = parse_args(get_args())
    data = dataset[0]

    train_mask = data.train_mask
    sens = data.sens_attrs.flatten()
    x = data.x

    index = 1

    binary = []  # continuous features
    for i in range(x.shape[1]):
        if len(set([round(x[j, i].item(), 4) for j in range(x.shape[0])])) == 2:
            binary.append(i)

    if debug:
        fig, ax = plt.subplots()
        ax.hist(x[sens == 0][:, index], bins=20, alpha=0.5, density=True, label="0")
        ax.hist(x[sens == 1][:, index], bins=20, alpha=0.5, density=True, label="1")
        ax.legend()
        plt.show()

    # normalize using mean/std of training set
    mean_0 = x[(sens == 0) & train_mask].mean(axis=0)
    std_0 = x[(sens == 0) & train_mask].std(axis=0)
    std_0[std_0 == 0] = 1

    mean_1 = x[(sens == 1) & train_mask].mean(axis=0)
    std_1 = x[(sens == 1) & train_mask].std(axis=0)
    std_1[std_1 == 0] = 1

    old_x = x.clone()
    x[sens == 0] = (x[sens == 0] - mean_0) / std_0
    x[sens == 1] = (x[sens == 1] - mean_1) / std_1
    x[:, binary] = old_x[:, binary]

    if debug:
        fig, ax = plt.subplots()
        ax.hist(x[sens == 0][:, index], bins=20, alpha=0.5, density=True, label="0")
        ax.hist(x[sens == 1][:, index], bins=20, alpha=0.5, density=True, label="1")
        ax.legend()
        plt.show()

    temp = torch.load(f"data/{dataset_name}/processed/{dataset_name}.pt")
    temp[0].x = x

    torch.save(temp, f"data/{dataset_name}/processed/{dataset_name}_modified.pt")
