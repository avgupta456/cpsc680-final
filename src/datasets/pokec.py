import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import ToDevice, ToUndirected

from src.utils import device, set_random_seed

set_random_seed(0)


def load_pokec_data(suffix: str = ""):
    df = pd.read_csv(f"data/pokec/region_job{suffix}.csv")
    columns = list(df.columns)

    sens_attr = "region"
    predict_attr = "I_am_working_in_field"

    columns.remove("user_id")
    columns.remove(sens_attr)
    columns.remove(predict_attr)

    features = df[columns].values

    # Normalize features to have mean 0 and std 1
    # TODO: very slight data leakage here, but it's not a big deal
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # -1 means missing value, 0 means no job, 1 means job
    labels = df[predict_attr].values
    labels = np.minimum(labels, 1)

    sens_attrs = df[sens_attr].values.reshape(-1, 1)

    idx = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt(
        f"data/pokec/region_job{suffix}_relationship.txt", dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    # Among the labeled nodes, randomly select 1000 validation, 1000 test, rest training
    labels_idx = np.where(labels >= 0)[0]
    val_test_idx = np.random.choice(labels_idx, size=2000, replace=False)
    val_idx = np.random.choice(val_test_idx, size=1000, replace=False)
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


pokec_z = PokecZDataset(pre_transform=ToUndirected(), transform=ToDevice(device))
pokec_n = PokecNDataset(pre_transform=ToUndirected(), transform=ToDevice(device))
