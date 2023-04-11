import torch

from src.node.argparser import get_args, parse_args
from src.node.model import MLP, train_mlps

if __name__ == "__main__":
    args = get_args()
    debug, dataset, dataset_name = parse_args(args)

    print(dataset, dataset_name)

    N = dataset.num_features

    encoder = MLP(N, [N], N, 0)
    decoder = MLP(N, [N], N, 0)
    classifier = MLP(N, [N // 2], 1, 0, True)

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=1e-3, weight_decay=1e-3
    )
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=1e-3, weight_decay=1e-3
    )
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=1e-3, weight_decay=1e-3
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

    print(classifier(encoder(dataset[0].x)))
    print(classifier(encoder(dataset[0].x)).mean())

    print(dataset[0].sens_attrs)
    print(dataset[0].sens_attrs.to(float).mean())
