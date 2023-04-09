import matplotlib.pyplot as plt
import torch

from src.argparser import get_args, parse_vanilla_args


def histogram(data_arr, label_arr, bins=100, density=True, alpha=0.5):
    _, ax = plt.subplots()
    for data, label in zip(data_arr, label_arr):
        ax.hist(
            data,
            bins=bins,
            density=density,
            alpha=alpha,
            label=label,
        )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    _, _, dataset, _, _, _ = parse_vanilla_args(args)

    plot = True

    node_model = torch.load("models/CreditDataset_Node_GCNConv(16,1)_Adam_100.pt")
    edge_model = torch.load("models/CreditDataset_Edge_GCNConv(16)_Adam_100.pt")

    data = dataset[0]
    edge_prob_1 = edge_model(data.x, data.edge_index, data.edge_index).sigmoid()
    edge_prob_2 = edge_model.decode(
        node_model.embedding(data.x, data.edge_index), data.edge_index
    ).sigmoid()

    if plot:
        histogram(
            [edge_prob_1.detach().numpy(), edge_prob_2.detach().numpy()],
            ["Edge Model", "Node Model"],
        )

    sens_attrs = data.sens_attrs.flatten().to(int)
    edge_homophily = sens_attrs[data.edge_index[0]] == sens_attrs[data.edge_index[1]]

    if plot:
        # stacked histogram of edge prob 1 for homophily and non-homophily
        histogram(
            [
                edge_prob_1.detach().numpy()[edge_homophily],
                edge_prob_1.detach().numpy()[~edge_homophily],
            ],
            ["Homophily", "Non-Homophily"],
        )

        # stacked histogram of edge prob 2 for homophily and non-homophily
        histogram(
            [
                edge_prob_2.detach().numpy()[edge_homophily],
                edge_prob_2.detach().numpy()[~edge_homophily],
            ],
            ["Homophily", "Non-Homophily"],
        )

    edge_prob = edge_prob_1

    # Remove homophily edges with low edge probability
    edge_index = data.edge_index[:, ~(edge_homophily & (edge_prob < 0.7))]

    print(
        f"Removed {data.edge_index.shape[1] - edge_index.shape[1]} edges (out of {data.edge_index.shape[1]})"
    )

    dataset_name = args.dataset
    folder_name = "pokec" if "pokec" in dataset_name else dataset_name

    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].edge_index = edge_index

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_modified.pt")
