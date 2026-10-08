"""Training function for the EncoderDecoder model."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from minifigures_model.metrics import metric_class_balanced_f1
from minifigures_model.model import EncoderDecoder


def train(
    dataloader: DataLoader,
    model: EncoderDecoder,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
) -> tuple[list[float], list[float]]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        Training data loader
    model : EncoderDecoder
        The model to train
    optimizer : torch.optim.Optimizer
        The optimizer
    loss_fn : nn.Module
        The loss function
    epoch : int
        Current epoch number (for display)

    Returns
    -------
    tuple[list[float], list[float]]
        Lists of losses and f1 scores per batch
    """
    model.train()
    losses, f1_scores = [], []
    with tqdm(total=len(dataloader), desc=f"Training epoch={epoch}") as pbar:
        for batch in dataloader:
            pred = model(batch["image"])
            loss = loss_fn(pred, batch["label"])
            f1 = metric_class_balanced_f1(pred, batch["label"])

            losses.append(loss.item())
            f1_scores.append(f1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {"loss": sum(losses) / len(losses), "f1": sum(f1_scores) / len(f1_scores)}
            )
            pbar.update()
    return losses, f1_scores
