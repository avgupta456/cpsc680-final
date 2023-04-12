import torch

from src.node.argparser import get_args, parse_args
from src.node.model import MLP, train_mlps

if __name__ == "__main__":
    args = get_args()
    debug, dataset, dataset_name = parse_args(args)

    N = dataset.num_features

    M = min(N, 32)

    mid = (N + M) // 2

    encoder = MLP(N, [mid], M, 0)
    decoder = MLP(M, [mid], N, 0)
    classifier = MLP(M, [M // 4], 1, 0, True)

    print(N, M, mid)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0, weight_decay=1e-3)
    decoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-3,
        weight_decay=0,
    )
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=1e-3, weight_decay=5e-2
    )

    torch.autograd.set_detect_anomaly(True)

    train_mlps(
        encoder,
        decoder,
        classifier,
        dataset_name,
        dataset,
        encoder_optimizer,
        decoder_optimizer,
        classifier_optimizer,
        1000,
        debug,
    )

    pred = classifier(encoder(dataset.x))
    actual = dataset[0].sens_attrs.to(float)

    print(actual[pred > 0.5].mean())
    print(actual[pred < 0.5].mean())
