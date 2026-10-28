import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	logits = np.array(logits)
	logits = np.exp(logits)
	logits /= np.sum(logits)

	y_true = np.zeros_like(logits)
	y_true[target] = 1

	return (logits - y_true).tolist()