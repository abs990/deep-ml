import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    idx = np.argsort(y_scores)[::-1]
    y_true = y_true[idx]
    y_scores = y_scores[idx]

    thresholds = np.array([float('inf')])
    thresholds = np.unique(np.append(thresholds, y_scores))[::-1]

    fpr_list = []
    tpr_list = []

    for thr in thresholds:
        TP = np.sum((y_true == 1) & (y_scores >= thr))
        FN = np.sum((y_true == 1) & (y_scores < thr))
        FP = np.sum((y_true == 0) & (y_scores >= thr))
        TN = np.sum((y_true == 0) & (y_scores < thr))

        tpr = 0
        fpr = 0

        if TP > 0 or FN > 0:
            tpr = 1.0 * TP / (TP + FN)

        if FP > 0 or TN > 0:
            fpr  = 1.0 * FP / (FP + TN)

        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    return fpr_list, tpr_list
