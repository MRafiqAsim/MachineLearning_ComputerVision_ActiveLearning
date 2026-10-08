"""Pydantic basemodels."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Prediction base model."""

    prediction: dict[str, float] = Field(
        ...,
        title="prediction",
        description="Dictionary containing the predicted probability for each class",
        example={
            "alien": 0.07,
            "angry": 0.07,
            "cape": 0.04,
            "facial hair": 0.15,
            "glasses": 0.23,
            "happy": 0.11,
            "hat": 0.11,
            "helmet": 0.41,
            "human": 0.60,
            "robot": 0.15,
        },
    )
