import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
		return np.corrcoef(X, rowvar=False)
	
	return np.array([
		[np.corrcoef(X[:, i], Y[:, j])[0, 1] for j in range(Y.shape[1])] 
		for i in range(X.shape[1])
		])