"""Merge new Label Studio exports into train.json ONLY — val/test stay frozen."""

import glob
import json
from pathlib import Path

datasets_f = Path("data/data/datasets")

# Load the frozen splits
train = json.load(open(datasets_f / "train.json"))
val = json.load(open(datasets_f / "val.json"))
test = json.load(open(datasets_f / "test.json"))

# Load the latest Label Studio export
export_files = sorted(glob.glob("data/data/dataset_labeled*.json"))
assert export_files, "No dataset_labeled*.json export found"
new_labels = json.load(open(export_files[-1]))
print(f"Export: {Path(export_files[-1]).name} with {len(new_labels)} labels")

added, corrected = 0, 0
for tag, labels in new_labels.items():
    if tag in val:
        if val[tag] != labels:  # label correction of a val image
            val[tag] = labels
            corrected += 1
    elif tag in test:
        if test[tag] != labels:
            test[tag] = labels
            corrected += 1
    elif tag not in train or train[tag] != labels:
        if tag not in train:
            added += 1
        train[tag] = labels  # new label (or train correction) -> train only

# Safety: splits must remain disjoint
assert not (set(train) & set(val)), "train/val overlap!"
assert not (set(train) & set(test)), "train/test overlap!"
assert not (set(val) & set(test)), "val/test overlap!"

json.dump(train, open(datasets_f / "train.json", "w"), indent=2)
json.dump(val, open(datasets_f / "val.json", "w"), indent=2)
json.dump(test, open(datasets_f / "test.json", "w"), indent=2)

print(f"Added {added} new labels to TRAIN (val/test untouched, {corrected} in-place corrections)")
print(f"Sizes now: train={len(train)}  val={len(val)}  test={len(test)}")
