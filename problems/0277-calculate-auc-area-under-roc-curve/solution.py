import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    idx = np.argsort(y_scores)[::-1]
    y_true = y_true[idx]
    y_scores = y_scores[idx]

    thresholds = np.unique(y_scores)[::-1]

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

    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += 0.5 * (tpr_list[i] + tpr_list[i - 1]) * (fpr_list[i] - fpr_list[i - 1])

    return auc