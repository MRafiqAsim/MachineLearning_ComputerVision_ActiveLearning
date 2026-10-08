"""Minifigure Vision REST API."""

import logging
import threading
from importlib.metadata import version

import coloredlogs
from fastapi import FastAPI

from minifigures_api.routers import data_router, fetch_model, predict_router
from minifigures_api.routers.predict import build_embedding_index

app = FastAPI(
    title="Minifigure Vision API",
    description="Multi-label attribute prediction and visual similarity search for minifigure images.",
    version=version("minifigures-app"),
    docs_url="/",  # Put docs under default URL
)


@app.on_event("startup")
def startup_event() -> None:
    """Run API startup events."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    # Add coloredlogs' coloured StreamHandler to the root logger.
    coloredlogs.install()

    # Load model on startup (prevent cold starts)
    _ = fetch_model()

    # Precompute embedding index for similarity search (in background)
    threading.Thread(target=build_embedding_index, daemon=True).start()


# Specify the different endpoint routers
app.include_router(data_router, prefix="/data")
app.include_router(predict_router, prefix="/predict")
