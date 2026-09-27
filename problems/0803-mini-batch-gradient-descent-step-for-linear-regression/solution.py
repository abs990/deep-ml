import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    # Given
    batch_size = len(batch_indices)
    X_batch = X[batch_indices, :]
    y_batch = y[batch_indices]

    # Gradient descent update
    y_diff = np.dot(X_batch, weights) + bias - y_batch
    bias_grad = 2.0 * np.mean(y_diff)
    weights_grad = (2.0 / batch_size ) * np.dot(X_batch.T, y_diff)
    bias -= lr * bias_grad
    weights -= lr * weights_grad

    return np.append(weights, bias)