"""Tests for the public-dataset preparation script."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "labeling"))

from prepare_dataset import convert, read_csv, slugify


def test_slugify() -> None:
    """Character names become lowercase, dash-separated tags."""
    assert slugify("Spider-Man") == "spider-man"
    assert slugify("Captain America ") == "captain-america"
    assert slugify("R2-D2 / C-3PO") == "r2-d2-c-3po"


def test_read_csv_accepts_windows_1252(tmp_path: Path) -> None:
    """The dataset's metadata file contains Windows-1252 apostrophes."""
    path = tmp_path / "metadata.csv"
    path.write_bytes("class_id,minifigure_name\n1,Rey\x92s friend\n".encode("latin-1"))
    assert read_csv(path) == [{"class_id": "1", "minifigure_name": "Rey’s friend"}]


def test_convert_writes_tagged_pngs_and_catalog(tmp_path: Path) -> None:
    """Images are renamed per character and listed in the catalogue."""
    source = tmp_path / "source"
    (source / "marvel" / "0001").mkdir(parents=True)
    for name in ("001.jpg", "002.jpg"):
        Image.new("RGB", (8, 8)).save(source / "marvel" / "0001" / name)
    (source / "metadata.csv").write_text(
        "class_id,lego_ids,lego_names,minifigure_name\n1,[1],['Spider Mech'],SPIDER-MAN\n"
    )
    (source / "index.csv").write_text(
        "path,class_id\nmarvel/0001/001.jpg,1\nmarvel/0001/002.jpg,1\n"
    )
    (source / "test.csv").write_text("path,class_id\n")

    catalog = convert(source, tmp_path / "out")

    assert sorted(catalog) == ["spider-man_001", "spider-man_002"]
    assert catalog["spider-man_001"]["lego_set"] == "Spider Mech"
    assert (tmp_path / "out" / "minifigures" / "spider-man_002.png").is_file()
