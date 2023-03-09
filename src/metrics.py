import torch

import numpy as np

from src.datasets.pokec import pokec_n, pokec_z
from src.vanilla_gnn import VanillaGNN


def fair_metric(labels, sens, output, idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0.5).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(
        sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
        - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
    )

    return parity, equality


def eval_model(data, model):
    model.eval()
    output = model(data)
    labels = data.y
    sens = data.sens_attrs.flatten()

    idx = data.test_mask
    acc = (output[idx].squeeze() > 0.5).eq(labels[idx]).sum().item() / idx.sum().item()
    parity, equality = fair_metric(labels, sens, output, idx)

    print("Test set results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Parity: {parity:.4f}")
    print(f"Equality: {equality:.4f}")


if __name__ == "__main__":
    dataset = pokec_z
    model: VanillaGNN = torch.load("models/pokec_z_16.pt")

    eval_model(dataset[0], model)
