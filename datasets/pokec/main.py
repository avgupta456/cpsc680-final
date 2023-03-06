import numpy as np
import pandas as pd
import torch

from torch_geometric.data import InMemoryDataset, Data


def load_pokec_data(suffix: str = ""):
    df = pd.read_csv(f"data/pokec/region_job{suffix}.csv")
    columns = list(df.columns)

    sens_attr = "region"
    predict_attr = "I_am_working_in_field"

    columns.remove("user_id")
    columns.remove(sens_attr)
    columns.remove(predict_attr)

    features = df[columns].values
    labels = df[predict_attr].values
    sens_attrs = df[sens_attr].values.reshape(-1, 1)

    idx = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_unordered = np.genfromtxt(
        f"data/pokec/region_job{suffix}_relationship.txt", dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edge_unordered.flatten())), dtype=int
    ).reshape(edge_unordered.shape)

    data = Data(x=features, edge_index=edges.T, y=labels, sens_attrs=sens_attrs)
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


if __name__ == "__main__":
    pokec_z = PokecZDataset()
    print(pokec_z.data)

    pokec_n = PokecNDataset()
    print(pokec_n.data)
