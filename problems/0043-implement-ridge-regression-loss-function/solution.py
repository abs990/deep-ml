import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	return alpha * np.sum(w ** 2) + np.sum((y - X @ w) ** 2) / X.shape[0]
