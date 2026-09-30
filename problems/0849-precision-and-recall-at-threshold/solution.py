import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    # predictions
    y_pred = [int(score >= threshold) for score in y_scores]
    
    # metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = 0.0
    if tp > 0 or fp > 0:
        precision = 1.0 * tp / (tp + fp)

    recall = 0.0
    if tp > 0 or fn > 0:
        recall = 1.0 * tp / (tp + fn)

    return [round(precision, 4), round(recall, 4)]
