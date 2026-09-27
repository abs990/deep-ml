import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	theta = np.dot(np.linalg.pinv(np.array(X)), np.array(y))
	return np.round(theta, 4).tolist()