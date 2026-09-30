import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	X = np.column_stack((np.ones(X.shape[0]), X)) # bias
	m, n = X.shape

	# init weights
	w = np.zeros(n)

	# init losses
	losses = []

	for _ in range(iterations):
		pred = np.dot(X, w)
		prob = 1.0 / (1 + np.exp(-pred))
		
		# loss - TBD
		loss = -1.0 * np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
		losses.append(round(loss, 4))

		# weight update
		grad = np.dot(X.T, prob - y)
		w -= learning_rate * grad

	return np.round(w, 4).tolist(), losses