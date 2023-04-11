import tqdm

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, dropout=0, sigmoid=False
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
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


def run_encoder(
    encoder, decoder, classifier, data, mask, classifier_weight, optimizer=None
):
    if optimizer:
        optimizer.zero_grad()
        encoder.train()
        decoder.train()
        classifier.train()
    else:
        encoder.eval()
        decoder.eval()
        classifier.eval()

    z = encoder(data.x[mask])
    x_hat = decoder(z)
    y_hat = classifier(z)

    decoder_pred = x_hat
    decoder_target = data.x[mask]
    # Decoder wants to reconstruct the input
    decoder_loss = F.mse_loss(decoder_pred, decoder_target)

    classifier_pred = y_hat
    classifier_target = data.sens_attrs[mask].to(torch.float32)
    # Classifier wants to predict the sensitive attributes, encoder wants to hide them
    classifier_loss = F.binary_cross_entropy(
        classifier_pred, classifier_target, weight=classifier_weight
    )

    encoder_loss = decoder_loss - classifier_loss
    # encoder_loss = decoder_loss + classifier_loss

    if optimizer:
        encoder_loss.backward()
        optimizer.step()

    return encoder_loss, decoder_loss, classifier_loss


def run_decoder(encoder, decoder, data, mask, optimizer=None):
    if optimizer:
        optimizer.zero_grad()
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    z = encoder(data.x[mask])
    x_hat = decoder(z)

    decoder_pred = x_hat
    decoder_target = data.x[mask]
    # Decoder wants to reconstruct the input
    decoder_loss = F.mse_loss(decoder_pred, decoder_target)

    if optimizer:
        decoder_loss.backward()
        optimizer.step()

    return decoder_loss


def run_classifier(encoder, classifier, data, mask, classifier_weight, optimizer=None):
    if optimizer:
        optimizer.zero_grad()
        encoder.train()
        classifier.train()
    else:
        encoder.eval()
        classifier.eval()

    z = encoder(data.x[mask])
    y_hat = classifier(z.detach())

    classifier_pred = y_hat
    classifier_target = data.sens_attrs[mask].to(torch.float32)
    # Classifier wants to predict the sensitive attributes, encoder wants to hide them
    classifier_loss = F.binary_cross_entropy(
        classifier_pred, classifier_target, weight=classifier_weight
    )

    if optimizer:
        classifier_loss.backward()
        optimizer.step()

    classifier_correct = classifier_pred.round().eq(classifier_target).sum().item()
    classifier_count = mask.sum().item()
    classifier_acc = classifier_correct / classifier_count

    return (classifier_loss, classifier_acc)


def train_mlps(
    encoder,
    decoder,
    classifier,
    dataset_name,
    dataset,
    encoder_optimizer,
    decoder_optimizer,
    classifier_optimizer,
    epochs,
    debug,
):
    print(f"Training {dataset_name} encoder-decoder-classifier model...")

    data = dataset[0]
    best_models = None

    sens = data.sens_attrs
    pos_count = sens[data.train_mask].sum().item()
    neg_count = sens[data.train_mask].shape[0] - pos_count
    pos_weight = (pos_count + neg_count) / pos_count / 2
    neg_weight = (pos_count + neg_count) / neg_count / 2
    train_classifier_weight = torch.where(sens[data.train_mask], pos_weight, neg_weight)
    val_classifier_weight = torch.where(sens[data.val_mask], pos_weight, neg_weight)
    test_classifier_weight = torch.where(sens[data.test_mask], pos_weight, neg_weight)

    iterator = range(epochs) if debug else tqdm.tqdm(range(epochs))
    for epoch in iterator:
        # Training
        train_encoder_loss, _, _ = run_encoder(
            encoder,
            decoder,
            classifier,
            data,
            data.train_mask,
            train_classifier_weight,
            encoder_optimizer,
        )
        train_decoder_loss = run_decoder(
            encoder, decoder, data, data.train_mask, decoder_optimizer
        )
        train_classifier_loss, train_classifier_acc = run_classifier(
            encoder,
            classifier,
            data,
            data.train_mask,
            train_classifier_weight,
            classifier_optimizer,
        )

        # Validation
        val_encoder_loss, val_decoder_loss, _ = run_encoder(
            encoder, decoder, classifier, data, data.val_mask, val_classifier_weight
        )
        (val_classifier_loss, val_classifier_acc) = run_classifier(
            encoder, classifier, data, data.val_mask, val_classifier_weight
        )

        if debug:
            print(
                f"Epoch: {epoch}, Train Loss: ({train_encoder_loss:.4f}, {train_decoder_loss:.4f}, {train_classifier_loss:.4f}), Train Acc: {train_classifier_acc:.4f}, Val Loss: ({val_encoder_loss:.4f}, {val_decoder_loss:.4f}, {val_classifier_loss:.4f}), Val Acc: {val_classifier_acc:.4f}"
            )

        if best_models is None or val_encoder_loss < best_models[0]:
            best_models = (
                val_encoder_loss,
                encoder.state_dict(),
                decoder.state_dict(),
                classifier.state_dict(),
            )

    print()

    encoder.load_state_dict(best_models[1])
    decoder.load_state_dict(best_models[2])
    classifier.load_state_dict(best_models[3])

    test_encoder_loss, test_decoder_loss, _ = run_encoder(
        encoder, decoder, classifier, data, data.test_mask, test_classifier_weight
    )
    test_classifier_loss, test_classifier_acc = run_classifier(
        encoder, classifier, data, data.test_mask, test_classifier_weight
    )

    print(
        f"Test Loss: ({test_encoder_loss:.4f}, {test_decoder_loss:.4f}, {test_classifier_loss:.4f}), Test Acc: {test_classifier_acc:.4f}"
    )
    print()

    torch.save(encoder, f"models/{dataset_name}_encoder.pt")
    torch.save(decoder, f"models/{dataset_name}_decoder.pt")
    torch.save(classifier, f"models/{dataset_name}_classifier.pt")
