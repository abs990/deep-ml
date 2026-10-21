import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    n_samples, n_features = data.shape
    alphas = np.zeros(n_samples)
    beta = 0.0

    #kernels
    def linear(x, z):
        return x @ z

    def rbf(x, z):
        return np.exp(-0.5 * np.sum((x - z)**2) / sigma**2)

    f = linear if kernel == 'linear' else rbf

    # compute kernel values between data points
    K = np.empty((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = f(data[i], data[j])

    # pegasos kernel svm
    for t in range(1, 1 + iterations):
        eta = 1.0 / (lambda_val * t)
        for i in range(n_samples):
            g_i = beta + np.sum(alphas * labels * K[:, i])
            if labels[i] * g_i < 1:
                alphas[i] += eta * (1 - lambda_val * alphas[i])
                beta += eta * labels[i]
            else:
                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas, beta