import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	f_x = np.array(f(x))

	M = len(f_x)
	N = len(x)
	
	J = np.zeros((M, N))

	for idx in range(N):
		xh = x.copy()
		xh[idx] += h

		f_xh = np.array(f(xh))

		J[:, idx] = (f_xh - f_x) / h

	return J.tolist()