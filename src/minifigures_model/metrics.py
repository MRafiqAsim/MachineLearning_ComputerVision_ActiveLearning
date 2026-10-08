"""Metric functions for model evaluation."""

import torch


def metric_precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Precision is the fraction of relevant instances among the retrieved instances."""
    if pred.sum() == 0:
        return 1.0
    if target.sum() == 0:
        return 0.0
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    return float(tp / max(tp + fp, 1e-8))


def metric_recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Recall is the fraction of relevant instances that were actually retrieved."""
    if target.sum() == 0:
        return 1.0
    if pred.sum() == 0:
        return 0.0
    tp = (pred * target).sum()
    fn = (target * (1 - pred)).sum()
    return float(tp / max(tp + fn, 1e-8))


def metric_f1_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """The f1 score is the harmonic mean of precision and recall."""
    p = metric_precision(pred=pred, target=target)
    r = metric_recall(pred=pred, target=target)
    return 2 * (p * r) / max(p + r, 1e-8)


def metric_class_balanced_f1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate the class balanced f1 score."""
    f1s = [
        metric_f1_score(
            pred=(logits[:, i] >= 0.0).to(torch.float64),
            target=(target[:, i] >= 0.5).to(torch.float64),
        )
        for i in range(target.shape[1])
    ]
    return torch.mean(torch.tensor(f1s)).item()
