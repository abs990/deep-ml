import numpy as np

def gauss_seidel(A, b, n, x_ini=None):	
	num_coeffs = A.shape[0]
	
	# inital guess
	if x_ini is None:
		x_ini = np.zeros_like(b)

	# iterate
	for _ in range(n):
		x_next = np.zeros_like(x_ini)
		for i in range(num_coeffs):
			i_next = i + 1
			x_next[i] = (b[i] - np.sum(A[i, :i] * x_next[:i]) - np.sum(A[i, i_next:] * x_ini[i_next:])) / A[i, i]
		x_ini = x_next

	return x_ini