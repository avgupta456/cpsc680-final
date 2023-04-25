import copy
import tqdm

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, dropout=0, sigmoid=False
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if len(hidden_channels) == 0:
            self.layers.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(torch.nn.Linear(in_channels, hidden_channels[0]))
            for i in range(1, len(hidden_channels)):
                self.layers.append(
                    torch.nn.Linear(hidden_channels[i - 1], hidden_channels[i])
                )
            self.layers.append(torch.nn.Linear(hidden_channels[-1], out_channels))

        self.dropout = dropout
        self.sigmoid = sigmoid

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](x)

        if self.sigmoid:
            x = x.sigmoid()

        return x


def run_encoder_classifier(
    encoder,
    classifier,
    x,
    sens,
    classifier_weight,
    l1_rate,
    encoder_optimizer=None,
    classifier_optimizer=None,
):
    if encoder_optimizer:
        encoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        encoder.train()
        classifier.train()
    else:
        encoder.eval()
        classifier.eval()

    x_hat = encoder(x)
    sens_hat = classifier(x_hat)

    decoder_pred = x_hat
    decoder_target = x
    # Decoder wants to reconstruct the input
    decoder_loss = F.mse_loss(decoder_pred, decoder_target)

    classifier_pred = sens_hat
    classifier_target = sens
    # Classifier wants to predict the sensitive attributes, encoder wants to hide them
    classifier_loss = F.binary_cross_entropy(
        classifier_pred, classifier_target, weight=classifier_weight
    )

    encoder_loss = decoder_loss - classifier_loss

    # Encoder gets L1 penalty to encourage sparsity (1 to 1 mapping)
    l1_penalty, N_params = 0, 0
    for param in encoder.parameters():
        l1_penalty += torch.abs(param).sum()
        N_params += param.numel()
    encoder_loss += l1_rate * l1_penalty / N_params

    if encoder_optimizer:
        encoder.requires_grad_(True)
        classifier.requires_grad_(False)
        # Encoder minimizes decoder loss, maximizes classifier loss
        encoder_loss.backward(retain_graph=True)
        encoder.requires_grad_(False)
        classifier.requires_grad_(True)
        # Classifier minimizes classifier loss
        classifier_loss.backward()
        encoder.requires_grad_(True)
        encoder_optimizer.step()
        classifier_optimizer.step()

    classifier_correct = classifier_pred.round().eq(classifier_target).sum().item()
    classifier_count = classifier_target.shape[0]
    classifier_acc = classifier_correct / classifier_count

    return encoder_loss, decoder_loss, classifier_loss, classifier_acc


def train_mlps(
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
):
    print(f"Training {dataset_name} encoder-predictor-classifier model...")

    data = dataset[0]
    best_models = None

    if estimate_sens_attrs:
        # Use predicted sensitive attributes (can be inaccurate)
        train_mask = data.all_train_mask
        sens_model = torch.load(f"models/{dataset_name}_sens_attr.pt")
        sens = sens_model(data.x, data.edge_index).unsqueeze(-1).detach()
    else:
        # Use ground truth sensitive attributes (fewer labels)
        train_mask = data.train_mask
        sens = data.sens_attrs.to(torch.float)

    pos_count = sens[train_mask].sum().item()
    neg_count = sens[train_mask].shape[0] - pos_count
    pos_weight = (pos_count + neg_count) / pos_count / 2
    neg_weight = (pos_count + neg_count) / neg_count / 2
    train_classifier_weight = pos_weight * sens[train_mask] + neg_weight * (
        1 - sens[train_mask]
    )
    val_classifier_weight = pos_weight * sens[data.val_mask] + neg_weight * (
        1 - sens[data.val_mask]
    )
    test_classifier_weight = pos_weight * sens[data.test_mask] + neg_weight * (
        1 - sens[data.test_mask]
    )

    iterator = range(epochs) if debug else tqdm.tqdm(range(epochs))
    for epoch in iterator:
        # Training
        (
            train_encoder_loss,
            train_decoder_loss,
            train_classifier_loss,
            train_classifier_acc,
        ) = run_encoder_classifier(
            encoder,
            classifier,
            data.x[train_mask],
            sens[train_mask],
            train_classifier_weight,
            l1_rate,
            encoder_optimizer,
            classifier_optimizer,
        )

        # Validation
        (
            val_encoder_loss,
            val_decoder_loss,
            val_classifier_loss,
            val_classifier_acc,
        ) = run_encoder_classifier(
            encoder,
            classifier,
            data.x[data.val_mask],
            data.sens_attrs.to(torch.float)[data.val_mask],
            val_classifier_weight,
            l1_rate,
        )

        if debug:
            print(
                f"Epoch: {epoch}, Train Loss: ({train_encoder_loss:.4f}, {train_decoder_loss:.4f}, {train_classifier_loss:.4f}), Train Acc: ({train_classifier_acc:.4f}), Val Loss: ({val_encoder_loss:.4f}, {val_decoder_loss:.4f}, {val_classifier_loss:.4f}), Val Acc: ({val_classifier_acc:.4f})"
            )

        if best_models is None or val_decoder_loss < best_models[0]:
            best_models = (
                val_decoder_loss,
                copy.deepcopy(encoder.state_dict()),
                copy.deepcopy(classifier.state_dict()),
            )

    print()

    (
        test_encoder_loss,
        test_decoder_loss,
        test_classifier_loss,
        test_classifier_acc,
    ) = run_encoder_classifier(
        encoder,
        classifier,
        data.x[data.test_mask],
        data.sens_attrs.to(torch.float)[data.test_mask],
        test_classifier_weight,
        l1_rate,
    )

    print(
        f"Test Loss: ({test_encoder_loss:.4f}, {test_decoder_loss:.4f}, {test_classifier_loss:.4f}), Test Acc: ({test_classifier_acc:.4f})"
    )
    print()

    torch.save(encoder, f"models/{dataset_name}_encoder.pt")
    torch.save(classifier, f"models/{dataset_name}_classifier.pt")
