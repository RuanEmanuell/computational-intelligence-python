from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

SEED = 42
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 8
EXPECTED_LOCAL_CLASSES = 5
LR = 1e-3

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int) -> nn.Module:
    return SimpleCNN(num_classes=num_classes)


def make_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )


def load_local_dataset(dataset_dir: Path):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if len(class_dirs) != EXPECTED_LOCAL_CLASSES:
        print(
            f"[INFO] Dataset local encontrado, mas eram esperadas {EXPECTED_LOCAL_CLASSES} classes "
            f"e foram encontradas {len(class_dirs)}. Usando dataset de fallback."
        )
        return None

    full_ds = datasets.ImageFolder(root=dataset_dir, transform=make_transform())
    class_names = full_ds.classes

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Usando dataset local.")
    print(f"[INFO] Classes: {class_names}")

    return train_loader, val_loader, val_loader, class_names


def filter_by_class(dataset, allowed_classes: set[int]):
    indices = [i for i, y in enumerate(dataset.targets) if int(y) in allowed_classes]
    return Subset(dataset, indices)


def load_fallback_fashion_mnist():
    transform = make_transform()
    selected_classes = {0, 1, 2, 3, 4}

    train_full = datasets.FashionMNIST(root="./data_cache", train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root="./data_cache", train=False, download=True, transform=transform)

    train_filtered = filter_by_class(train_full, selected_classes)
    test_filtered = filter_by_class(test_full, selected_classes)

    train_size = int(0.8 * len(train_filtered))
    val_size = len(train_filtered) - train_size
    train_ds, val_ds = random_split(
        train_filtered,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_filtered, batch_size=BATCH_SIZE, shuffle=False)

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat"]

    print("[INFO] Usando dataset de fallback Fashion-MNIST (classes 0-4).")
    print(f"[INFO] Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoca {epoch + 1}/{EPOCHS} "
            f"- perda: {train_loss:.4f} - acuracia: {train_acc:.4f} "
            f"- val_perda: {val_loss:.4f} - val_acuracia: {val_acc:.4f}"
        )

    return history


def plot_training_curves(history: dict[str, list[float]], output_path: Path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_conf_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def collect_predictions(model: nn.Module, dataset: DataLoader, device: torch.device):
    y_true_batches = []
    y_pred_batches = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataset:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true_batches.append(y_batch.numpy())
            y_pred_batches.append(y_pred)

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    return y_true, y_pred


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    loss, acc = run_epoch(model, loader, criterion, None, device)
    return loss, acc


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    dataset_dir = base_dir / "dataset"
    local_dir = data_dir if data_dir.exists() else dataset_dir

    loaded = load_local_dataset(local_dir)
    if loaded is None:
        train_loader, val_loader, test_loader, class_names = load_fallback_fashion_mnist()
    else:
        train_loader, val_loader, test_loader, class_names = loaded

    model = build_model(num_classes=len(class_names)).to(device)

    history = train_model(model, train_loader, val_loader, device)

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\n[RESULTADO] Perda no teste: {test_loss:.4f}")
    print(f"[RESULTADO] Acuracia no teste: {test_acc:.4f}")

    model_path = base_dir / "cnn_model.pth"
    curves_path = base_dir / "training_curves.png"
    cm_path = base_dir / "confusion_matrix.png"

    torch.save(model.state_dict(), model_path)
    plot_training_curves(history, curves_path)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    plot_conf_matrix(y_true, y_pred, class_names, cm_path)

    print(f"[RESULTADO] Modelo salvo: {model_path.name}")
    print(f"[RESULTADO] Grafico salvo: {curves_path.name}")
    print(f"[RESULTADO] Grafico salvo: {cm_path.name}")


if __name__ == "__main__":
    main()
