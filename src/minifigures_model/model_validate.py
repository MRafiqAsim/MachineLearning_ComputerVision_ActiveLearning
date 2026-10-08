"""Validation function for the EncoderDecoder model."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from minifigures_model.metrics import metric_class_balanced_f1
from minifigures_model.model import EncoderDecoder


def validate(
    dataloader: DataLoader, model: EncoderDecoder, loss_fn: nn.Module, epoch: int
) -> dict[str, float]:
    """
    Validate the model on the validation set.

    Parameters
    ----------
    dataloader : DataLoader
        Validation data loader
    model : EncoderDecoder
        The model to validate
    loss_fn : nn.Module
        The loss function
    epoch : int
        Current epoch number (for display)

    Returns
    -------
    dict[str, float]
        Dictionary with 'loss' and 'f1' keys
    """
    model.eval()
    test_loss, test_f1 = [], []
    with torch.no_grad(), tqdm(total=len(dataloader), desc=f"Validating epoch={epoch}") as pbar:
        for batch in dataloader:
            pred = model(batch["image"])
            test_loss += [loss_fn(pred, batch["label"]).item()] * batch["label"].shape[0]
            test_f1 += [metric_class_balanced_f1(pred, batch["label"])] * batch["label"].shape[0]
            pbar.set_postfix(
                {"loss": sum(test_loss) / len(test_loss), "f1": sum(test_f1) / len(test_f1)}
            )
            pbar.update()
    return {"loss": sum(test_loss) / len(test_loss), "f1": sum(test_f1) / len(test_f1)}
