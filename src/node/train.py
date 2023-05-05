import torch

from src.utils import device
from src.node.argparser import get_args, parse_args
from src.node.model import MLP, train_mlps

if __name__ == "__main__":
    args = get_args()
    (
        debug,
        dataset,
        dataset_name,
        estimate_sens_attrs,
        encoder_lr,
        encoder_l1_rate,
        classifier_lr,
        classifier_weight_decay,
        epochs,
    ) = parse_args(args)

    N = dataset.num_features

    print("N", N)

    encoder = MLP(N, [N], N, 0).to(device)
    classifier = MLP(N, [8, 8], 1, 0.25, True).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    l1_rate = encoder_l1_rate

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=classifier_lr, weight_decay=classifier_weight_decay
    )

    torch.autograd.set_detect_anomaly(True)

    train_mlps(
        encoder,
        classifier,
        dataset_name,
        dataset,
        encoder_optimizer,
        classifier_optimizer,
        epochs,
        l1_rate,
        estimate_sens_attrs,
        debug,
    )

    pred = classifier(encoder(dataset[0].x))
    actual = dataset[0].sens_attrs.to(float)

    print("Avg Sens Attr Pred (True):", actual[pred > 0.5].mean())
    print("Avg Sens Attr Pred (False):", actual[pred < 0.5].mean())

    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].x = encoder(dataset[0].x)

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_node.pt")
