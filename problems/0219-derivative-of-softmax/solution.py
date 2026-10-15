import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	x = np.array(x)
	
	# softmax
	x = np.exp(x)
	x = x / np.sum(x)

	# Jacobian
	J = np.diag(x) - (x[:, None] @ x[None, :])

	return J.tolist()