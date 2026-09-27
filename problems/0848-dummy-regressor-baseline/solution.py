import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    y_train = np.array(y_train)
    pred = 0
    if strategy == 'mean':
        pred = np.mean(y_train)
    elif strategy == 'median':
        pred = np.median(y_train)
    elif strategy == 'quantile' and quantile and 0 <= quantile <= 1:
        pred = np.quantile(y_train, quantile)
    elif strategy == 'constant' and constant:
        pred = constant
    else:
        raise ValueError("Invalid")
    pred = float(pred)
    return [pred] * n_test
