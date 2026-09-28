import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	z = np.dot(X, weights) + bias # regression
	z = np.clip(z, -500, 500) # prevent overflow
	prob = 1.0 / (1 + np.exp(-z)) # calculate probabilities
	return np.where(prob >= 0.5, 1, 0)