import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here

    X = np.array(X)

    # norm
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm_sq = np.matmul(X_norm, X_norm.T)
    X_norm_sq[X_norm_sq == 0.0] = 1 # prevent zero div error for zero L2 norm case

    return np.matmul(X, X.T) / X_norm_sq