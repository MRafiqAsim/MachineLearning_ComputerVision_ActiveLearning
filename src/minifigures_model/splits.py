"""Frozen train / validation / test splits.

The splits are created once from ``data/data/dataset.json`` and saved to
``data/data/datasets/{train,val,test}.json``. Every later run loads them instead of
re-splitting, and active-learning rounds add new labels to ``train.json`` only
(``src/labeling/merge_labels_train_only.py``). Validation and test therefore stay the
same images across rounds, so models from different rounds are compared fairly.
"""

import json
import random
from pathlib import Path

from minifigures_model.constants import get_data_folder

SPLITS = ("train", "val", "test")


def get_splits_folder() -> Path:
    """Folder holding the frozen split files."""
    return get_data_folder() / "datasets"


def create_splits(
    dataset: dict[str, list[str]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Shuffle the labeled tags reproducibly and split them into train, val and test."""
    keys = sorted(dataset)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_fraction))
    n_test = max(1, int(len(keys) * test_fraction))
    return {
        "val": {k: dataset[k] for k in keys[:n_val]},
        "test": {k: dataset[k] for k in keys[n_val : n_val + n_test]},
        "train": {k: dataset[k] for k in keys[n_val + n_test :]},
    }


def load_or_create_splits(
    folder: Path | None = None, seed: int = 42
) -> dict[str, dict[str, list[str]]]:
    """Load the frozen splits, creating and saving them from dataset.json on the first run."""
    folder = folder or get_splits_folder()
    if all((folder / f"{name}.json").exists() for name in SPLITS):
        return {name: json.loads((folder / f"{name}.json").read_text()) for name in SPLITS}

    dataset = json.loads((folder.parent / "dataset.json").read_text())
    splits = create_splits(dataset, seed=seed)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        (folder / f"{name}.json").write_text(json.dumps(data, indent=2))
    return splits
