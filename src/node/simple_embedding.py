import pandas as pd
import torch

from fairlearn.preprocessing import CorrelationRemover

from src.eval.main import eval_model
from src.vanilla.node_gnn import VanillaNode
from src.node.argparser import get_args, parse_args

if __name__ == "__main__":
    args = get_args()
    (debug, dataset, dataset_name, _, _, _, _, _, _) = parse_args(args)

    data = dataset[0]
    train_mask = data.train_mask
    sens = data.sens_attrs

    model: VanillaNode = torch.load(f"models/{dataset_name}.pt")

    embeddings = model.embedding(data.x, data.edge_index).detach().numpy()

    embeddings_df = pd.DataFrame(
        embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
    )

    embeddings_df["sens_attr"] = sens

    cr = CorrelationRemover(sensitive_feature_ids=["sens_attr"], alpha=1)

    cr.fit(embeddings_df.iloc[train_mask])
    # cr.fit(embeddings_df)

    embeddings_transform = cr.transform(embeddings_df)
    embeddings_transform = torch.tensor(embeddings_transform).float()

    mlp = torch.nn.Sequential(
        torch.nn.Linear(embeddings_transform.shape[1], 1),
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-3, weight_decay=1e-3)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = mlp(embeddings_transform[train_mask]).squeeze()
        loss = loss_fn(y_pred, data.y[train_mask].float())
        loss.backward()
        optimizer.step()

        y_pred_test = mlp(embeddings_transform[data.test_mask]).squeeze()
        loss_test = loss_fn(y_pred_test, data.y[data.test_mask].float())
        # print(f"Epoch {epoch} | Loss: {loss} | Test Loss: {loss_test}")

    eval_model(data, mlp(embeddings_transform).squeeze().sigmoid())
