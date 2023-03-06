import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomNodeSplit

from src.utils import set_random_seed

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

    # TODO: invesigate what labels mean, why there are 5 or 6 of them
    # Ranges from -1 to 3 for pokec_z, -1 to 4 for pokec_n
    labels = df[predict_attr].values + 1
    sens_attrs = df[sens_attr].values.reshape(-1, 1)

    idx = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt(
        f"data/pokec/region_job{suffix}_relationship.txt", dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    data = Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edges.T),
        y=torch.from_numpy(labels),
        sens_attrs=torch.from_numpy(sens_attrs),
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


pokec_z = PokecZDataset(
    transform=RandomNodeSplit(num_val=1000, num_test=1000),
)

pokec_n = PokecNDataset(
    transform=RandomNodeSplit(num_val=1000, num_test=1000),
)

print(pokec_z[0].y.unique())
print(pokec_n[0].y.unique())
