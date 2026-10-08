"""Put the annotations of label-studio from the target directory in a simple json format.

Keys: image tags, values: list of annotated classes.
Only keep annotated samples (ignore skipped ones).
The file is saved as dataset_labeled.json in the destination directory.
If the file already exists, a timestamp is added to the filename.
"""

import json
from datetime import datetime
from pathlib import Path


def export_label_studio_annotations_to_simple_json(source_dir: Path, destination_dir: Path) -> None:
    """Convert label-studio annotations to json."""
    dataset = {}
    for path in source_dir.iterdir():
        with open(path) as f:
            metadata = json.load(f)

        # Only save non-skipped samples
        if not metadata["was_cancelled"]:
            labels = metadata["result"][0]["value"]["choices"]
            image_path: str = metadata["task"]["data"]["image"]
            image_tag = image_path.split("/")[-1][:-4]  # Remove .png extension

            dataset[image_tag] = labels

    # Save dataset
    file_name = "dataset_labeled"
    dest_path = destination_dir / f"{file_name}.json"
    if dest_path.exists():
        file_name = f"{file_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"  # noqa: DTZ005
        dest_path = destination_dir / f"{file_name}.json"

    with open(dest_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Exported {len(dataset)} annotations to {dest_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    label_studio_annotations_dir = project_root / "data" / "annotations"
    save_dir = project_root / "data" / "data"

    export_label_studio_annotations_to_simple_json(label_studio_annotations_dir, save_dir)
