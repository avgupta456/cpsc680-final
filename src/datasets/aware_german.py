import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from src.utils import device, set_random_seed

set_random_seed(0)


def load_aware_german_data():
    df = pd.read_csv("data/german/german.csv")
    columns = list(df.columns)

    # change "Gender" attributes from "Female", "Male" to 1,0 respectively
    df = df.replace("Female",1)
    df = df.replace("Male",0)

    sens_attr = "Gender"
    predict_attr = "GoodCustomer"

    columns.remove(predict_attr)
    columns.remove("OtherLoansAtStore")
    columns.remove("PurposeOfLoan")

    features = df[columns].values

    # Normalize features to have mean 0 and std 1
    # TODO: very slight data leakage here, but it's not a big deal
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # -1, 0 --> 0, 1
    labels = df[predict_attr].values
    labels[labels == -1] = 0

    sens_attrs = df[sens_attr].values

    idx = np.arange(features.shape[0], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt("data/german/german_edges.txt", dtype=float).astype(
        int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    # Among the labeled nodes, randomly select 100 validation, 100 test, rest training
    labels_idx = np.arange(labels.shape[0], dtype=int)
    val_test_idx = np.random.choice(labels_idx, size=200, replace=False)
    val_idx = np.random.choice(val_test_idx, size=100, replace=False)
    test_idx = np.setdiff1d(val_test_idx, val_idx)
    train_idx = np.setdiff1d(labels_idx, val_test_idx)

    train_mask = np.zeros(labels.shape, dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros(labels.shape, dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros(labels.shape, dtype=bool)
    test_mask[test_idx] = True

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


class AwareGermanDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        super().__init__("data/aware_german", transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "aware_german.pt"

    def process(self):
        data: Data = load_aware_german_data()
        data = self.collate([data])

        torch.save(data, self.processed_paths[0])

aware_german = AwareGermanDataset(transform=T.Compose([T.ToDevice(device), T.ToUndirected()]))