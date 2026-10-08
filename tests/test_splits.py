"""Frozen train / validation / test splits."""

import json
from pathlib import Path

from minifigures_model.splits import SPLITS, create_splits, load_or_create_splits

DATASET = {f"tag_{i:02d}": ["human"] for i in range(40)}


def test_create_splits_is_disjoint_complete_and_reproducible() -> None:
    """Every tag lands in exactly one split, the same way on every run."""
    splits = create_splits(DATASET, seed=7)
    assert {len(splits[s]) for s in ("val", "test")} == {6}
    assert sum(len(splits[s]) for s in SPLITS) == len(DATASET)
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])
    assert create_splits(DATASET, seed=7) == splits


def test_splits_are_created_once_and_then_frozen(tmp_path: Path) -> None:
    """New labels in dataset.json never move images between validation and test."""
    folder = tmp_path / "datasets"
    (tmp_path / "dataset.json").write_text(json.dumps(DATASET))
    first = load_or_create_splits(folder)
    assert all((folder / f"{name}.json").exists() for name in SPLITS)

    # More labels arrive later: the saved splits are reused, not recomputed
    (tmp_path / "dataset.json").write_text(json.dumps({**DATASET, "tag_new": ["robot"]}))
    assert load_or_create_splits(folder) == first
