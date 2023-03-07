import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from src.datasets.pokec import pokec_z, pokec_n


class VanillaGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, 1)
        self.conv2 = GCNConv(hidden_channels, out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.log_softmax(dim=-1)

        return x


def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def val_gnn(model, data):
    model.eval()
    out = model(data)
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    y_pred = out[data.val_mask].max(dim=1)[1]
    y_true = data.y[data.val_mask].max(dim=1)[1]
    correct = y_pred.eq(y_true).sum().item()
    count = data.val_mask.sum().item()
    acc = correct / count
    return loss, acc


def test_gnn(model, data):
    model.eval()
    out = model(data)
    loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])
    y_pred = out[data.test_mask].max(dim=1)[1]
    y_true = data.y[data.test_mask].max(dim=1)[1]
    correct = y_pred.eq(y_true).sum().item()
    count = data.test_mask.sum().item()
    acc = correct / count
    return loss, acc


def train_model(dataset, dataset_name, epochs=200):
    print(f"Training {dataset_name} model...")

    data = dataset[0]
    model = VanillaGNN(
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=16,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_model = None
    for epoch in range(epochs):
        loss = train_gnn(model, data, optimizer)
        val_loss, val_acc = val_gnn(model, data)
        print(
            f"Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc = test_gnn(model, data)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
    print()

    # save model
    torch.save(model.state_dict(), f"models/{dataset_name}.pt")


if __name__ == "__main__":
    train_model(pokec_z, "pokec_z", 50)
    train_model(pokec_n, "pokec_n", 50)
