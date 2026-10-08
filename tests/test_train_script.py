"""Tests for the training script's data split."""

from minifigures_model.train_script import split_dataset


def test_split_is_disjoint_complete_and_reproducible() -> None:
    """The split covers every tag once and does not change between runs."""
    dataset = {f"tag_{i:02d}": ["human"] for i in range(20)}
    train, val = split_dataset(dataset, val_fraction=0.2, seed=7)

    assert len(val) == 4
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(dataset)
    assert split_dataset(dataset, val_fraction=0.2, seed=7) == (train, val)
