import torchmetrics

from src.argparser import get_args, parse_metric_args
from src.metrics.fairness import get_parity, get_equality
from src.metrics.bias import get_attribute_bias


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
    parity = get_parity(labels, sens, output, idx)
    equality = get_equality(labels, sens, output, idx)
    attribute_bias = get_attribute_bias(data.x, sens)

    print("Test set results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Parity: {parity:.4f}")
    print(f"Equality: {equality:.4f}")
    print()

    print("Graph bias results:")
    print(f"Attribute bias: {attribute_bias:.4f}")
    print()


if __name__ == "__main__":
    args = get_args()
    dataset, model = parse_metric_args(args)

    eval_model(dataset[0], model)
