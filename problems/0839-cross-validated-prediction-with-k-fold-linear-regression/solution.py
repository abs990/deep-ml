import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    # Your code here
    X = np.array(X)
    y = np.array(y)

    n, _ = X.shape

    X = np.column_stack([X, np.ones(n)]) #augment for intercept

    # compute split indices
    base, extra = divmod(n, k) # base fold size, extra samples to distribute
    folds, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(list(range(start, start + size)))
        start += size

    preds = []

    for i, val_idx in enumerate(folds):
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

        # Index the original arrays
        X_t, y_t = X[train_idx], y[train_idx]
        X_v, y_v = X[val_idx], y[val_idx]

        W = np.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)

        preds += (X_v @ W).tolist()

    return preds