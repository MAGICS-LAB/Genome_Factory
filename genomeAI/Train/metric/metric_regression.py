import numpy as np
import torch
import sklearn.metrics


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    """
    Calculate the MSE and MAE with sklearn.
    Exclude any labels set to -100 (used for padding).
    """
    valid_mask = labels != -100  # Exclude padding tokens if any
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    mse = sklearn.metrics.mean_squared_error(valid_labels, valid_predictions)
    mae = sklearn.metrics.mean_absolute_error(valid_labels, valid_predictions)
    return {"mse": mse, "mae": mae}

def preprocess_logits_for_metrics(logits, _):
    """
    For huggingface trainer: preprocess logits for metrics calculation.
    For regression, simply squeeze the last dimension.
    """
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.squeeze(-1)

def compute_metrics(eval_pred):
    """
    Compute regression metrics using sklearn.
    """
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)