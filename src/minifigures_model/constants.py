"""Utilisation functions."""

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_TAG = os.getenv("MODEL_TAG", "my_model")


@lru_cache
def get_root_folder() -> Path:
    """Load in the root folder."""
    current = Path(__file__).resolve()
    # Search upwards for the first folder that contains a directory named 'data'
    for parent in current.parents:
        if (parent / "data").is_dir():
            return parent / "data"
    # Fallback: create a data folder in the current directory
    return current.parent / "data"


@lru_cache
def get_data_folder() -> Path:
    """Load in the data folder."""
    folder = get_root_folder() / "data"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


@lru_cache
def get_models_folder() -> Path:
    """Load in the model folder."""
    folder = get_root_folder() / "models"
    folder.mkdir(exist_ok=True, parents=True)
    return folder
