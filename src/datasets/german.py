import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from src.utils import device


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

    train_mask = np.zeros(labels.shape, dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros(labels.shape, dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros(labels.shape, dtype=bool)
    test_mask[test_idx] = True

    # Normalize features to have mean 0 and std 1
    mean, std = features[train_mask].mean(axis=0), features[train_mask].std(axis=0)
    features = (features - mean) / std

    data = Data(
        x=torch.from_numpy(features).float(),
        edge_index=torch.from_numpy(edges.T).long(),
        y=torch.from_numpy(labels).float(),
        sens_attrs=torch.from_numpy(sens_attrs).bool(),
        train_mask=torch.from_numpy(train_mask).bool(),
        val_mask=torch.from_numpy(val_mask).bool(),
        test_mask=torch.from_numpy(test_mask).bool(),
    )

    return data


class GermanDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/german", transform, pre_transform, pre_filter)
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
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/german", transform, pre_transform, pre_filter)
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


class GermanModifiedDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/german", transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "german_modified.pt"

    def process(self):
        pass


german = GermanDataset(transform=T.Compose([T.ToDevice(device), T.ToUndirected()]))
german_aware = GermanAwareDataset(
    transform=T.Compose([T.ToDevice(device), T.ToUndirected()])
)

try:
    german_modified = GermanModifiedDataset(
        transform=T.Compose([T.ToDevice(device), T.ToUndirected()])
    )
except FileNotFoundError:
    german_modified = german

german_link_pred = GermanDataset(
    transform=T.Compose(
        [
            T.ToDevice(device),
            T.ToUndirected(),
            T.RandomLinkSplit(
                num_val=0.1,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
            ),
        ]
    ),
)
