"""Create label-studio tasks with active learning scores.

A task basically corresponds to an image with optional metadata such as prediction scores.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from config import HOST, PORT
from label_studio_sdk import Client
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import transforms

from minifigures_model.constants import get_data_folder, get_models_folder
from minifigures_model.model import EncoderDecoder
from minifigures_model.utils import pil_to_torch, resize

# Same normalization as training (ImageNet values for EfficientNet-B0)
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# --- Paths ---
DATA_DIR = get_data_folder()  # e.g. .../data/data  (where images + dataset.json live)
IMAGE_DIR = DATA_DIR / "minifigures"
DATASET_JSON = DATA_DIR / "dataset.json"


def add_prediction_scores_to_label_studio(
    tags_scores: dict[str, float], model_version: str, token: str, project_id: int
) -> None:
    """Add prediction score attribute to label-studio tasks.

    Make sure the model_version does not exist yet.
    The project_id can be retrieved by clicking on the project in the UI
    and looking at the number after /projects/ in the url.
    """
    ls = Client(url=f"{HOST}:{PORT}", api_key=token)
    project = ls.get_project(project_id)

    assert model_version not in project.get_model_versions(), (
        "The model_version you provided, already exists. Choose a different one."
    )

    print("Setting prediction scores in label-studio...")
    for task in project.get_tasks():
        image_path = Path(task["storage_filename"])
        tag = image_path.stem

        if tag in tags_scores:
            project.create_prediction(
                task["id"], result=[], score=tags_scores[tag], model_version=model_version
            )


def add_active_learning_scores(model_version: str, token: str, project_id: int, k: int = 5) -> None:
    """Add loss of labeled image to its k nearest neighbor images as prediction score.

    1. Sort the images in the trainset from highest to lowest loss
    2. For every trainset image: retrieve the k most similar images (embeddingwise) from the unlabeled dataset.
    3. Add prediction score attribute in the existing label-studio tasks.
    """
    # Get all labeled tags
    tags_all_label = get_tags_labeled()

    # Get worst predictions wrt loss and rank
    print("Computing loss on the trainset...\n")
    losses_tags = get_loss_trainset()
    losses, tags = list(losses_tags.values()), list(losses_tags.keys())
    sorter = np.argsort(np.array(losses))[::-1]
    tags = [tags[idx] for idx in sorter]
    losses = [losses[idx] for idx in sorter]

    # Load embeddings and put in datastructure for efficient searching
    print("Initializing k nearest neighbors...\n")
    index = get_embeddings()
    embeddings_nolabel = np.array([index[tag] for tag in index if tag not in tags_all_label])
    tags_nolabel = [tag for tag in index if tag not in tags_all_label]
    embeddings_label = np.array([index[tag] for tag in tags])
    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn_model.fit(embeddings_nolabel)

    # Loop over all labeled images
    tags_nn_loss = {}
    for embedding, loss in tqdm(
        zip(embeddings_label, losses, strict=True),
        desc="Computing nearest neighbor for every train image...",
    ):
        knn = nn_model.kneighbors([embedding], return_distance=False)[0, :]
        for ind_knn in knn:
            tag_nn = tags_nolabel[ind_knn]
            tags_nn_loss[tag_nn] = round(loss, 4)

    # Update prediction score of nn unlabeled images with given model_version
    add_prediction_scores_to_label_studio(
        tags_nn_loss, model_version=model_version, token=token, project_id=project_id
    )


def get_embeddings() -> dict[str, list[float]]:
    """Get embeddings of all images (labeled + unlabeled) using the encoder.

    Return as a dictionary with key: image tag, value: embedding as a list of floats.
    """
    model = _load_model()
    model.eval()

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    embeddings = {}
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Extracting embeddings"):
            tag = img_path.stem
            img = pil_to_torch(Image.open(img_path))
            img = resize(img, resolution=model.resolution)
            img = normalize(img)
            img = img.unsqueeze(0)

            # Extract embedding from the encoder (before the decoder)
            embedding = model.encoder(img).squeeze(0).numpy()
            embeddings[tag] = embedding.tolist()

    print(f"Extracted embeddings for {len(embeddings)} images")
    return embeddings


def get_loss_trainset() -> dict[str, float]:
    """Get loss on each trainset image using the model.

    Return as a dictionary with key: image tag, value: loss.
    """
    model = _load_model()
    model.eval()

    with open(DATASET_JSON) as f:
        dataset = json.load(f)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses_per_tag = {}

    with torch.no_grad():
        for tag, labels in tqdm(dataset.items(), desc="Computing per-sample loss"):
            img_path = IMAGE_DIR / f"{tag}.png"
            if not img_path.exists():
                continue

            img = pil_to_torch(Image.open(img_path))
            img = resize(img, resolution=model.resolution)
            img = normalize(img)
            img = img.unsqueeze(0)

            logits = model(img)
            target = torch.FloatTensor([[1.0 if c in labels else 0.0 for c in model.classes]])

            loss = loss_fn(logits, target).mean().item()
            losses_per_tag[tag] = loss

    print(f"Computed loss for {len(losses_per_tag)} training samples")
    return losses_per_tag


def get_tags_labeled() -> list[str]:
    """Get all tags of images that are already labeled."""
    with open(DATASET_JSON) as f:
        dataset = json.load(f)
    return list(dataset.keys())


def _load_model() -> EncoderDecoder:
    """Load the latest trained model."""
    models_dir = get_models_folder()
    model_tags = [p.name for p in models_dir.iterdir() if p.is_dir()]
    if not model_tags:
        msg = f"No trained models found in {models_dir}. Train a model first."
        raise FileNotFoundError(msg)

    tag = sorted(model_tags)[-1]
    print(f"Loading model: {tag}")
    return EncoderDecoder.load(tag)


if __name__ == "__main__":
    # Add active learning scores
    import os

    # A new version name for every active-learning round, e.g. model_v1, model_v2, ...
    model_version = os.environ.get("AL_MODEL_VERSION", "model_v1")
    # Label Studio > Account & Settings > Access Token
    token = os.environ["LABEL_STUDIO_TOKEN"]
    # The number after /projects/ in the Label Studio project URL
    project_id = int(os.environ.get("LABEL_STUDIO_PROJECT_ID", "1"))
    add_active_learning_scores(model_version, token, project_id, k=5)
