"""Download the public LEGO Minifigures dataset and convert it to this project's layout.

Source: "LEGO Minifigures" by the TinySets development team (Yaroslav Isaienkov),
https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification, licensed under
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Images are converted to PNG and
renamed; no other changes are made.

Output (in ``data/data``):

- ``minifigures/<tag>.png``  one image per tag, e.g. ``spider-man_001.png``
- ``catalog.json``           tag -> character name, LEGO set and source path
- ``dataset.json``           tag -> attribute labels; created empty, then filled by labeling
                             the images in Label Studio (see the README)

Usage::

    uv run python src/labeling/prepare_dataset.py            # download from Kaggle
    uv run python src/labeling/prepare_dataset.py --source DIR  # use an extracted copy
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

from PIL import Image

KAGGLE_DATASET = "ihelon/lego-minifigures-classification"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "data"


def slugify(name: str) -> str:
    """Lowercase, dash-separated version of a character name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def download() -> Path:
    """Download the dataset with kagglehub (public dataset, no account needed)."""
    import kagglehub  # noqa: PLC0415 - only needed when downloading

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file; the dataset's metadata uses Windows-1252 apostrophes."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return list(csv.DictReader(text.splitlines()))


def read_metadata(source: Path) -> dict[str, dict[str, str]]:
    """Map class_id to character name and LEGO set."""
    classes = {}
    for row in read_csv(source / "metadata.csv"):
        sets = ast.literal_eval(row["lego_names"])
        classes[row["class_id"]] = {
            "name": row["minifigure_name"].title(),
            "lego_set": sets[0] if sets else "",
        }
    return classes


def convert(source: Path, destination: Path) -> dict[str, dict[str, str]]:
    """Copy every image as PNG under a readable tag and return the catalogue."""
    classes = read_metadata(source)
    image_dir = destination / "minifigures"
    image_dir.mkdir(parents=True, exist_ok=True)

    catalog: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for index_file in ("index.csv", "test.csv"):
        for row in read_csv(source / index_file):
            info = classes[row["class_id"]]
            slug = slugify(info["name"])
            counts[slug] = counts.get(slug, 0) + 1
            tag = f"{slug}_{counts[slug]:03d}"
            Image.open(source / row["path"]).convert("RGB").save(image_dir / f"{tag}.png")
            catalog[tag] = {**info, "source": row["path"]}
    return catalog


def main() -> None:
    """Prepare the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="Extracted dataset folder (skips download)")
    args = parser.parse_args()

    source = args.source or download()
    catalog = convert(source, DATA_DIR)

    with open(DATA_DIR / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    dataset_json = DATA_DIR / "dataset.json"
    if not dataset_json.exists():
        dataset_json.write_text("{}\n")

    print(
        f"Prepared {len(catalog)} images of {len(set(c['name'] for c in catalog.values()))} "
        f"minifigures in {DATA_DIR}"
    )
    print("Next: label a first batch of images in Label Studio (see the README).")


if __name__ == "__main__":
    main()
