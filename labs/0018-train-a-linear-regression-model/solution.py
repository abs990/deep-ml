import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.

    n_samples, n_features = X.shape
    
    num_iterations = 1000
    lr = 0.01

    for _ in range(num_iterations):
        preds = X @ W + b
        errors = preds - y
        grad_W = (2 / n_samples) * (X.T @ errors)
        grad_b = (2 / n_samples) * np.sum(errors)
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b
