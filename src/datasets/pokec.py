import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from src.utils import device


def load_pokec_data(suffix, aware):
    df = pd.read_csv(f"data/pokec/region_job{suffix}.csv")
    columns = list(df.columns)

    sens_attr = "region"
    predict_attr = "I_am_working_in_field"

    columns.remove("user_id")
    columns.remove(sens_attr)
    columns.remove(predict_attr)

    features = df[columns].values

    # -1 means missing value, 0 means no job, 1 means job
    labels = df[predict_attr].values
    labels = np.minimum(labels, 1)

    sens_attrs = df[sens_attr].values.reshape(-1, 1)

    if aware:
        features = np.concatenate([features, sens_attrs], axis=1)

    idx = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt(
        f"data/pokec/region_job{suffix}_relationship.txt", dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    """
    NOTE: Identical setup to FairGNN paper (seed 20)
    Model is only unfair with so few (500) training nodes
    """

    random.seed(20)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[: min(int(0.5 * len(label_idx)), 500)]
    idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
    idx_test = label_idx[int(0.75 * len(label_idx)) :]

    train_mask = np.zeros(labels.shape, dtype=bool)
    train_mask[idx_train] = True

    val_mask = np.zeros(labels.shape, dtype=bool)
    val_mask[idx_val] = True

    test_mask = np.zeros(labels.shape, dtype=bool)
    test_mask[idx_test] = True

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


class PokecZDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/pokec", transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "pokec_z.pt"

    def process(self):
        data: Data = load_pokec_data(suffix="")
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


class PokecZAwareDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/pokec", transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "pokec_z_aware.pt"

    def process(self):
        data: Data = load_pokec_data(suffix="", aware=True)
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


class PokecNDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/pokec", transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "pokec_n.pt"

    def process(self):
        data: Data = load_pokec_data(suffix="_2")
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


class PokecNAwareDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/pokec", transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "pokec_n_aware.pt"

    def process(self):
        data: Data = load_pokec_data(suffix="_2", aware=True)
        data = self.collate([data])
        torch.save(data, self.processed_paths[0])


pokec_z = PokecZDataset(transform=T.Compose([T.ToDevice(device), T.ToUndirected()]))
pokec_z_aware = PokecZAwareDataset(
    transform=T.Compose([T.ToDevice(device), T.ToUndirected()])
)
pokec_z_modified = pokec_z_aware

pokec_z_link_pred = PokecZDataset(
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
    )
)

pokec_n = PokecNDataset(transform=T.Compose([T.ToDevice(device), T.ToUndirected()]))
pokec_n_aware = PokecNAwareDataset(
    transform=T.Compose([T.ToDevice(device), T.ToUndirected()])
)
pokec_n_modified = pokec_n_aware

pokec_n_link_pred = PokecNDataset(
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
    )
)
