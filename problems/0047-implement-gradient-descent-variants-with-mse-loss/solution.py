import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m, n = X.shape
    
    if method == 'batch':
        mse_const_term = 2.0 / m
        for _ in range(n_epochs):
            y_hat = np.dot(X, weights)
            grad = mse_const_term * np.dot(X.T, y_hat - y)
            weights -= learning_rate * grad
    elif method == 'stochastic':
        mse_const_term = 2.0
        for epoch in range(n_epochs):
            for sample_idx in range(m):
                X_sample = X[sample_idx]
                y_sample = y[sample_idx]
                y_hat = np.dot(X_sample, weights)
                grad = mse_const_term * np.dot(X_sample.T, y_hat - y_sample)
                weights -= learning_rate * grad
    else:
        num_batches = m // batch_size
        mse_const_term = 2.0 / batch_size
        for _ in range(n_epochs):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                X_sample = X[start_idx:end_idx]
                y_sample = y[start_idx:end_idx]                
                y_hat = np.dot(X_sample, weights)
                grad = mse_const_term * np.dot(X_sample.T, y_hat - y_sample)
                weights -= learning_rate * grad
    
    return weights