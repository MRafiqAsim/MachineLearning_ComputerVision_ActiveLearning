"""Train the EncoderDecoder model on the frozen train / validation splits.

The splits in ``data/data/datasets/`` are created from ``data/data/dataset.json`` on the first
run and reused afterwards (see ``minifigures_model.splits``), so every active-learning round is
validated on the same images.

Usage::

    uv run python src/minifigures_model/train_script.py --tag my_model --epochs 40

The best model (by validation F1) is saved to ``data/models/<tag>``. If the environment
variable ``MODELS_BUCKET`` is set, the model is also uploaded to ``s3://<bucket>/models/<tag>``
so the deploy pipeline can pick it up.
"""

import argparse
import json
import os
import subprocess

import torch
from torch import nn
from torch.utils.data import DataLoader

from minifigures_model.constants import DEFAULT_MODEL_TAG, get_data_folder, get_models_folder
from minifigures_model.dataset import MinifiguresDataset
from minifigures_model.model import EncoderDecoder
from minifigures_model.model_train import train
from minifigures_model.model_validate import validate
from minifigures_model.splits import load_or_create_splits


def run_training(
    tag: str, epochs: int = 40, lr: float = 3e-3, batch_size: int = 8, patience: int = 3
) -> float:
    """Train with cosine learning-rate decay and early stopping; return the best val F1."""
    data_dir = get_data_folder()
    with open(data_dir / "dataset.json") as f:
        dataset = json.load(f)
    if len(dataset) < 10:  # noqa: PLR2004
        msg = "Label at least 10 images in Label Studio before training (see the README)."
        raise SystemExit(msg)

    splits = load_or_create_splits()
    train_data, val_data = splits["train"], splits["val"]
    print(f"train: {len(train_data)}  val: {len(val_data)}  test (held out): {len(splits['test'])}")

    # One attribute list for every split, so label vectors line up with the model outputs
    classes = sorted({c for split in splits.values() for labels in split.values() for c in labels})
    image_dir = data_dir / "minifigures"
    train_loader = DataLoader(
        MinifiguresDataset(image_dir, train_data, classes=classes),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MinifiguresDataset(image_dir, val_data, classes=classes), batch_size=batch_size
    )

    model = EncoderDecoder(tag=tag, classes=classes)
    print(f"Classes: {classes}")

    # Only the decoder head is trained; the EfficientNet encoder stays frozen
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1, epochs_without_improvement = 0.0, 0
    for epoch in range(1, epochs + 1):
        train(train_loader, model, optimizer, loss_fn, epoch)
        scheduler.step()
        metrics = validate(val_loader, model, loss_fn, epoch)
        print(f"Epoch {epoch}: val_loss={metrics['loss']:.4f}  val_f1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1, epochs_without_improvement = metrics["f1"], 0
            model.save()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}")
                break

    model_dir = get_models_folder() / tag
    print(f"Best val F1: {best_f1:.4f}  model saved to {model_dir}")

    bucket = os.environ.get("MODELS_BUCKET")
    if bucket:
        s3_path = f"s3://{bucket}/models/{tag}"
        subprocess.run(["aws", "s3", "cp", str(model_dir), s3_path, "--recursive"], check=True)  # noqa: S603, S607
        print(f"Uploaded model to {s3_path}")
    return best_f1


def main() -> None:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description="Train the minifigure attribute classifier")
    parser.add_argument(
        "--tag", default=DEFAULT_MODEL_TAG, help="Model name (folder under data/models)"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience")
    args = parser.parse_args()
    run_training(
        tag=args.tag,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
