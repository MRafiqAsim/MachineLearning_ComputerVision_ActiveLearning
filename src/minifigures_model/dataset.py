"""Custom dataset for minifigures classification."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from minifigures_model.utils import pil_to_torch, resize


def get_transform_normalize() -> transforms.Compose:
    """Generate the image normalization function."""
    return transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MinifiguresDataset(Dataset):
    """Custom dataset for minifigures classification."""

    def __init__(
        self,
        data_f: Path,
        dataset: dict[str, list[str]],
        resolution: int = 256,
        classes: list[str] | None = None,
    ) -> None:
        """
        Initialise the dataset.

        Parameters
        ----------
        data_f : Path
            Path to the data (images) folder
        dataset : dict[str, list[str]]
            Dictionary containing the dataset and target data
        resolution : int
            Resolution to transform the images to
        classes : list[str] | None
            Attribute order for the label vectors; defaults to the attributes in ``dataset``.
            Pass the same list for every split so the vectors line up.
        """
        self.data_f = data_f
        self.keys, self.labels = zip(*dataset.items())
        self.classes = classes or sorted({x for y in self.labels for x in y})
        self.resolution = resolution
        self.transform_normalize = get_transform_normalize()

    def __len__(self) -> int:
        """Amount of images within the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load in the first next item from our dataset."""
        img = pil_to_torch(Image.open(self.data_f / f"{self.keys[idx]}.png"))
        img = resize(img, resolution=self.resolution)
        img = self.transform_normalize(img)
        label = torch.FloatTensor([x in self.labels[idx] for x in self.classes])
        return {"image": img, "label": label, "tag": self.keys[idx]}
