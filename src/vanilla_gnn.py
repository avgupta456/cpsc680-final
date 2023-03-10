import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from src.datasets.pokec import pokec_z, pokec_n


class VanillaGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, 1)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.sigmoid()

        return x.squeeze()


class VanillaGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, 1)
        self.conv2 = GATConv(hidden_channels, out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.sigmoid()

        return x.squeeze()


class VanillaSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.sigmoid()

        return x.squeeze()


class VanillaGIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GINConv(torch.nn.Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.sigmoid()

        return x.squeeze()


def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    mask = data.train_mask
    out = model(data)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)
    loss.backward()
    optimizer.step()

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count

    return loss, acc


def val_gnn(model, data):
    model.eval()
    mask = data.val_mask
    out = model(data)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count

    return loss, acc


def test_gnn(model, data):
    model.eval()
    mask = data.test_mask
    out = model(data)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count

    return loss, acc


def train_model(dataset, dataset_name, n_hidden, epochs):
    print(f"Training {dataset_name} model...")

    data = dataset[0]
    model = VanillaGCN(
        in_channels=dataset.num_features,
        hidden_channels=n_hidden,
        out_channels=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)

    best_model = None
    for epoch in range(epochs):
        train_loss, train_acc = train_gnn(model, data, optimizer)
        val_loss, val_acc = val_gnn(model, data)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc = test_gnn(model, data)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
    print()

    # save model
    torch.save(model, f"models/{dataset_name}_{n_hidden}.pt")


if __name__ == "__main__":
    train_model(pokec_z, "pokec_z", 16, 50)
    train_model(pokec_n, "pokec_n", 16, 50)
