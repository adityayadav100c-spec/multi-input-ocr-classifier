import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            nn.Linear(32 * 4 * 4, 32),
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


def evaluate_predictions(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-input handwritten digit classifier."
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive integers.")
    set_seed(args.seed)

    dataset = load_digits()
    images = (dataset.images / 16.0).astype(np.float32)
    labels = dataset.target.astype(np.int64)
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=args.seed,
        stratify=labels[train_val_idx],
    )

    raw_features = auxiliary_features(images)
    scaler = MinMaxScaler().fit(raw_features[train_idx])
    features = scaler.transform(raw_features).astype(np.float32)
    x = torch.tensor(images[:, None, :, :], dtype=torch.float32)
    f = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_data = TensorDataset(x[train_idx], f[train_idx], y[train_idx])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = MultiInputCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_images, batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_images, batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x[val_idx], f[val_idx]), y[val_idx]).item()
        train_loss = total_loss / len(train_idx)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss}
        )
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} — "
            f"train loss {train_loss:.4f}, validation loss {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        neural_predictions = model(x[test_idx], f[test_idx]).argmax(dim=1).numpy()

    image_features = images.reshape(len(images), -1)
    baseline = LogisticRegression(max_iter=500, random_state=args.seed)
    baseline.fit(image_features[train_idx], labels[train_idx])
    baseline_predictions = baseline.predict(image_features[test_idx])
    y_test = labels[test_idx]

    model_predictions = {
        "multi_input_cnn": neural_predictions,
        "logistic_regression_baseline": baseline_predictions,
    }
    metrics = {}
    for name, predictions in model_predictions.items():
        metrics[name] = evaluate_predictions(y_test, predictions)
        print(
            f"{name}: test accuracy={metrics[name]['accuracy']:.3%}, "
            f"macro F1={metrics[name]['macro_f1']:.3f}"
        )
        print(classification_report(y_test, predictions, zero_division=0))

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output / "multi_input_cnn.pt")
    (output / "preprocessing.json").write_text(
        json.dumps(
            {
                "feature_scale": scaler.scale_.tolist(),
                "feature_offset": scaler.min_.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savetxt(
        output / "test_predictions.csv",
        np.column_stack(
            [test_idx, y_test, neural_predictions, baseline_predictions]
        ),
        delimiter=",",
        header="dataset_index,true_label,multi_input_prediction,baseline_prediction",
        comments="",
        fmt="%d",
    )
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "scikit-learn load_digits (8x8 grayscale digits)",
                "seed": args.seed,
                "split_sizes": {
                    "train": len(train_idx),
                    "validation": len(val_idx),
                    "test": len(test_idx),
                },
                "epochs_run": len(history),
                "best_validation_loss": best_val_loss,
                "models": metrics,
                "limitations": [
                    "Digits only; not letter recognition or full-page OCR.",
                    "Small built-in dataset; results do not establish real-world accuracy.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    example_indices = test_idx[:10]
    figure, axes = plt.subplots(2, 5, figsize=(9, 4))
    for axis, sample_index in zip(axes.flat, example_indices):
        axis.imshow(images[sample_index], cmap="gray")
        prediction = int(neural_predictions[np.where(test_idx == sample_index)[0][0]])
        axis.set_title(f"true {labels[sample_index]} / pred {prediction}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=150)
    plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, name in zip(axes, model_predictions):
        matrix = np.asarray(metrics[name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        for row in range(10):
            for col in range(10):
                axis.text(col, row, matrix[row, col], ha="center", va="center")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    plt.figure(figsize=(7, 4))
    plt.plot(
        [item["epoch"] for item in history],
        [item["train_loss"] for item in history],
        label="train",
    )
    plt.plot(
        [item["epoch"] for item in history],
        [item["validation_loss"] for item in history],
        label="validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved model, evaluation, and visualizations to {output}")


if __name__ == "__main__":
    main()
