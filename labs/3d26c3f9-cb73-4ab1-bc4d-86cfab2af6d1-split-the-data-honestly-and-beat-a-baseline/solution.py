import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    m, n = X.shape

    # split the data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(m)

    train_cut = round(m * train_frac)
    val_cut = round(m * (train_frac + val_frac))

    train_idx = indices[:train_cut]
    val_idx = indices[train_cut:val_cut]
    test_idx = indices[val_cut:]

    train_X = X[train_idx, :]
    train_y = y[train_idx]
    val_X = X[val_idx, :]
    val_y = y[val_idx]
    test_X = X[test_idx, :]
    test_y = X[test_idx]

    # Columns: bias, x, x^2
    train_X_quad = np.column_stack((np.ones_like(train_X), train_X, train_X**2))
    val_X_quad = np.column_stack((np.ones_like(val_X), val_X, val_X**2))
    test_X_quad = np.column_stack((np.ones_like(test_X), test_X, test_X**2))

    # training setup
    m, n = train_X_quad.shape
    w = np.zeros(n)
    lr = 0.01
    iterations = 1000
    val_error = 10000

    # fit model as long as vaidation error declines
    for _ in range(iterations):
        y_hat = train_X_quad @ w
        grad = (2.0 / m) * (train_X_quad.T @ (y_hat - train_y))

        # adjust
        w -= lr * grad

        # check val error
        val_mae = np.mean(np.abs((val_X_quad @ w) - val_y))
        if val_mae > val_error:
            break
        val_error = val_mae

    test_predictions = test_X_quad @ w
    return test_predictions, train_idx, val_idx, test_idx