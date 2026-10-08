"""Endpoint for model predictions."""

import logging

import numpy as np
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from minifigures_api.routers.basemodels import Prediction
from minifigures_api.routers.utils import extract_images_from_files, fetch_model
from minifigures_model.constants import get_data_folder

router = APIRouter()
logger = logging.getLogger(__name__)

# Global embedding cache
import threading

_embedding_cache: dict[str, np.ndarray] = {}
_embedding_ready = False
_embedding_building = False
_embedding_lock = threading.Lock()


def build_embedding_index() -> None:
    """Precompute embeddings for all images into the global cache."""
    global _embedding_cache, _embedding_ready, _embedding_building  # noqa: PLW0603
    with _embedding_lock:
        if _embedding_ready or _embedding_building:
            return
        _embedding_building = True

    model = fetch_model()
    image_dir = get_data_folder() / "minifigures"
    image_files = sorted(image_dir.glob("*.png"))
    logger.info(f"Building embedding index for {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        tag = img_path.stem
        if tag.startswith("."):
            continue
        img = Image.open(img_path).convert("RGB")
        _embedding_cache[tag] = model.get_embedding(img).numpy()
        if (i + 1) % 500 == 0:
            logger.info(f"  Embedded {i + 1}/{len(image_files)} images")
    _embedding_ready = True
    logger.info(f"Embedding index built: {len(_embedding_cache)} images")


@router.post(
    "/image/",
    tags=["prediction"],
    response_model=Prediction,
    response_class=JSONResponse,
    responses={
        200: {"model": Prediction},
        404: {"content": {"application/json": {}}, "description": "Model Not Found"},
        415: {"content": {"application/json": {}}, "description": "Media type not valid"},
    },
)
def predict(file: UploadFile) -> Prediction:
    """Use the tagged model to predict over an image."""
    # Convert the UploadFile to a torch Tensor
    file = extract_images_from_files([file])[0]

    # Make a prediction
    model = fetch_model()
    prediction = model.predict(file)

    # Return the result
    return Prediction(prediction=prediction)


@router.get("/similar/", tags=["prediction"], response_class=JSONResponse)
def get_similar(tag: str, k: int = 5) -> dict:
    """Find the k most similar images to the given tag using cached embeddings."""
    if not _embedding_ready:
        return {"similar": [], "status": "building index, try again later"}

    if tag not in _embedding_cache:
        return {"similar": []}

    query_emb = _embedding_cache[tag]

    # Compute cosine similarity with all other images
    similarities = []
    for other_tag, other_emb in _embedding_cache.items():
        if other_tag == tag:
            continue
        sim = float(
            np.dot(query_emb, other_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(other_emb))
        )
        similarities.append((other_tag, float(sim)))

    # Sort by similarity (highest first) and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return {"similar": [{"tag": t, "score": s} for t, s in similarities[:k]]}
