import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.shared import transform, link_transform


def load_german_data(aware):
    df = pd.read_csv("data/german/german.csv")
    columns = list(df.columns)

    sens_attr = "Gender"
    predict_attr = "GoodCustomer"

    columns.remove(sens_attr)
    columns.remove(predict_attr)
    columns.remove("OtherLoansAtStore")
    columns.remove("PurposeOfLoan")

    features = df[columns].values

    # -1, 0 --> 0, 1
    labels = df[predict_attr].values
    labels[labels == -1] = 0

    sens_attrs = df[sens_attr].values.reshape(-1, 1)
    sens_attrs[sens_attrs == "Female"] = 1
    sens_attrs[sens_attrs == "Male"] = 0
    sens_attrs = sens_attrs.astype(int)

    if aware:
        features = np.concatenate([features, sens_attrs], axis=1)

    idx = np.arange(features.shape[0], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt("data/german/german_edges.txt", dtype=float).astype(
        int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    """
    NOTE: Identical setup to EDITS paper (seed 20)
    Model is only unfair with so few (100) training nodes
    """

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    all_train_idx = np.append(
        label_idx_0[: int(0.5 * len(label_idx_0))],
        label_idx_1[: int(0.5 * len(label_idx_1))],
    )
    train_idx = np.append(
        label_idx_0[: min(int(0.5 * len(label_idx_0)), 100 // 2)],
        label_idx_1[: min(int(0.5 * len(label_idx_1)), 100 // 2)],
    )
    val_idx = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
    )
    test_idx = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)) :],
        label_idx_1[int(0.75 * len(label_idx_1)) :],
    )

    all_train_mask = np.zeros(labels.shape, dtype=bool)
    all_train_mask[all_train_idx] = True

    train_mask = np.zeros(labels.shape, dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros(labels.shape, dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros(labels.shape, dtype=bool)
    test_mask[test_idx] = True

    # Normalize features to range [-1, 1], used in EDITS but leaks data
    # min_values = features.min(axis=0)
    # max_values = features.max(axis=0)
    # features = 2 * (features - min_values) / (max_values - min_values) - 1

    # Normalize features to have mean 0 and std 1
    mean, std = features[train_mask].mean(axis=0), features[train_mask].std(axis=0)
    features = (features - mean) / std

    data = Data(
        x=torch.from_numpy(features).float(),
        edge_index=torch.from_numpy(edges.T).long(),
        y=torch.from_numpy(labels).float(),
        sens_attrs=torch.from_numpy(sens_attrs).bool(),
        all_train_mask=torch.from_numpy(all_train_mask).bool(),
        train_mask=torch.from_numpy(train_mask).bool(),
        val_mask=torch.from_numpy(val_mask).bool(),
        test_mask=torch.from_numpy(test_mask).bool(),
    )

    return data


class GermanDataset(InMemoryDataset):
    def __init__(
        self, transform=None, pre_transform=None, pre_filter=None, filename=None
    ):
        super().__init__("data/german", transform, pre_transform, pre_filter)
        if filename is not None:
            self.data, self.slices = torch.load(filename)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "german.pt"

    def process(self):
        data: Data = load_german_data(False)
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


class GermanAwareDataset(InMemoryDataset):
    def __init__(
        self, transform=None, pre_transform=None, pre_filter=None, filename=None
    ):
        super().__init__("data/german", transform, pre_transform, pre_filter)
        if filename is not None:
            self.data, self.slices = torch.load(filename)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "german_aware.pt"

    def process(self):
        data: Data = load_german_data(True)
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


german = GermanDataset(transform=transform)
german_aware = GermanAwareDataset(transform=transform)
german_link_pred = GermanDataset(transform=link_transform)

try:
    german_node = GermanDataset(
        transform=transform, filename="./data/german/processed/german_node.pt"
    )
    german_aware_node = GermanDataset(
        transform=transform, filename="./data/german/processed/german_aware_node.pt"
    )
    german_node_link_pred = GermanDataset(
        transform=link_transform, filename="./data/german/processed/german_node.pt"
    )
except FileNotFoundError:
    german_node = german
    german_aware_node = german_aware
    german_node_link_pred = german_link_pred

try:
    german_edge = GermanDataset(
        transform=transform, filename="./data/german/processed/german_edge.pt"
    )
except FileNotFoundError:
    german_edge = german

try:
    german_node_edge = GermanDataset(
        transform=transform, filename="./data/german/processed/german_node_edge.pt"
    )
except FileNotFoundError:
    german_node_edge = german
