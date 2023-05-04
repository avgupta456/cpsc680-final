import pandas as pd
import torch

from fairlearn.preprocessing import CorrelationRemover
from sklearn.decomposition import PCA


from src.node.argparser import get_args, parse_args

if __name__ == "__main__":
    args = get_args()
    (debug, dataset, dataset_name, _, _, _, _, _, _) = parse_args(args)

    x = dataset[0].x
    train_mask = dataset[0].train_mask

    use_pca = True

    if use_pca:
        pca = PCA(n_components=20)
        pca.fit(x[train_mask])

        print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))

        x = pca.transform(x)

    x_df = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(x.shape[1])])
    x_df["sens_attr"] = dataset[0].sens_attrs

    cr = CorrelationRemover(sensitive_feature_ids=["sens_attr"], alpha=1)
    cr.fit(x_df.iloc[train_mask])
    # cr.fit(x_df)
    x_transform = cr.transform(x_df)
    x_transform = torch.tensor(x_transform).float()

    print("Original", x[0])
    print("Transformed", x_transform[0])

    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].x = x_transform

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_node.pt")
