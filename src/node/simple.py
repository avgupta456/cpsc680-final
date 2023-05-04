from fairlearn.preprocessing import CorrelationRemover
import pandas as pd
import torch

from src.node.argparser import get_args, parse_args

if __name__ == "__main__":
    args = get_args()
    (debug, dataset, dataset_name, _, _, _, _, _, _) = parse_args(args)

    train_mask = dataset[0].train_mask

    x = pd.DataFrame(
        dataset[0].x, columns=[f"feat_{i}" for i in range(dataset.num_features)]
    )
    x["sens_attr"] = dataset[0].sens_attrs

    # Can also look into Adversarial Fairness Classifier
    # https://fairlearn.org/v0.8/api_reference/fairlearn.adversarial.html#fairlearn.adversarial.AdversarialFairnessClassifier
    cr = CorrelationRemover(sensitive_feature_ids=["sens_attr"])
    cr.fit(x.iloc[train_mask])
    x_transform = cr.transform(x)
    x_transform = torch.tensor(x_transform).float()

    folder_name = dataset_name.split("_")[0]
    temp = torch.load(f"data/{folder_name}/processed/{dataset_name}.pt")
    temp[0].x = x_transform

    torch.save(temp, f"data/{folder_name}/processed/{dataset_name}_node.pt")
