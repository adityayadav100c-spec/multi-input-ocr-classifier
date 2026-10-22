import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).parent


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def auxiliary_features(images):
    return np.column_stack(
        [
            images.mean(axis=(1, 2)),
            images.std(axis=(1, 2)),
            images[:, :4, :].mean(axis=(1, 2)),
            images[:, 4:, :].mean(axis=(1, 2)),
        ]
    ).astype(np.float32)


class MultiInputCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 32),
            nn.ReLU(),
        )
        self.feature_branch = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 10)
        )

    def forward(self, image, features):
        combined = torch.cat(
            [self.image_branch(image), self.feature_branch(features)], dim=1
        )
        return self.classifier(combined)


def main():
    set_seed()
    dataset = load_digits()
    images = (dataset.images / 16.0).astype(np.float32)
    labels = dataset.target.astype(np.int64)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )

    features = auxiliary_features(images)
    scaler = MinMaxScaler().fit(features[train_idx])
    features = scaler.transform(features).astype(np.float32)
    x = torch.tensor(images[:, None, :, :], dtype=torch.float32)
    f = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_data = TensorDataset(x[train_idx], f[train_idx], y[train_idx])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = MultiInputCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 35
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_images, batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_images, batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs} — loss {total_loss / len(train_idx):.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(x[test_idx], f[test_idx]).argmax(dim=1)
        accuracy = (predictions == y[test_idx]).float().mean().item()
    print(f"Test accuracy ({len(test_idx)} held-out images): {accuracy:.3%}")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output / "multi_input_cnn.pt")
    figure, axes = plt.subplots(2, 5, figsize=(9, 4))
    for axis, index in zip(axes.flat, test_idx[:10]):
        axis.imshow(images[index], cmap="gray")
        axis.set_title(f"true {labels[index]} / pred {int(model(x[index:index+1], f[index:index+1]).argmax(1))}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=150)
    plt.close(figure)
    (output / "metrics.txt").write_text(
        f"seed={SEED}\ntrain_size={len(train_idx)}\ntest_size={len(test_idx)}\n"
        f"test_accuracy={accuracy:.6f}\n",
        encoding="utf-8",
    )
    print(f"Saved model, predictions, and metrics to {output}")


if __name__ == "__main__":
    main()
