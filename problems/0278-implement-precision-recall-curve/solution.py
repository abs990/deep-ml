import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    idx = np.argsort(y_scores)[::-1]

    thresholds = np.unique(y_scores[idx])[::-1]

    precisions = []
    recalls = []

    for thr in thresholds:
        TP = np.sum((y_true == 1) & (y_scores >= thr))
        FN = np.sum((y_true == 1) & (y_scores < thr))
        FP = np.sum((y_true == 0) & (y_scores >= thr))
        TN = np.sum((y_true == 0) & (y_scores < thr))

        precision = 1.0
        if TP > 0 or FP > 0:
            precision = 1.0 * TP / (TP + FP)

        recall = 0.0
        if TP > 0 or FN > 0:
            recall = 1.0 * TP / (TP + FN)

        precisions.append(float(precision))
        recalls.append(float(recall))

    return precisions, recalls, thresholds.tolist()