import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples, _ = X.shape

    for i in range(n_iter):
        sample_idx = i % n_samples

        grad = (2.0 * (np.dot(X[sample_idx], weights) - y[sample_idx])) * X[sample_idx].T
        weights -= learning_rate * grad

    return weights