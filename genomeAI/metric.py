
import numpy as np
import torch
import sklearn.metrics



def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    """
    Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
    """
    valid_mask = labels != -100  # Exclude padding tokens
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

def preprocess_logits_for_metrics(logits, _):
    """
    For huggingface trainer: reduce memory usage by returning only argmax of logits.
    """
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    """
    The compute_metrics used by huggingface trainer.
    """
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)