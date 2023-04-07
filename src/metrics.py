import numpy as np
import torchmetrics

from src.argparser import get_args, parse_metric_args


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
    output = model(data.x, data.edge_index)
    labels = data.y
    sens = data.sens_attrs.flatten()

    idx = data.test_mask
    acc = (output[idx].squeeze() > 0.5).eq(labels[idx]).sum().item() / idx.sum().item()
    auc = torchmetrics.functional.auroc(
        output[idx].squeeze(), labels[idx].to(int), task="binary"
    )
    f1 = torchmetrics.functional.f1_score(
        output[idx].squeeze(), labels[idx].to(int), task="binary"
    )
    parity, equality = fair_metric(labels, sens, output, idx)

    print("Test set results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Parity: {parity:.4f}")
    print(f"Equality: {equality:.4f}")


if __name__ == "__main__":
    args = get_args()
    dataset, model = parse_metric_args(args)

    eval_model(dataset[0], model)
