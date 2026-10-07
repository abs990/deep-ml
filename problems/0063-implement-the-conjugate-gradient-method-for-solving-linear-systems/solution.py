import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""

	# intialization
	if x0 is None:
		x0 = np.zeros_like(b)
	r = b - A @ x0
	p = r.copy()

	for _ in range(n):
		Ap = A @ p
		r2 = np.sum(r ** 2)
		
		alpha = r2 / (p.T @ Ap)
		x_next = x0 + alpha * p
		r_next = r - alpha * Ap

		x0 = x_next

		if np.sum(np.abs(r_next)) < tol:
			break

		beta = np.sum(r_next ** 2) / r2
		p_next = r_next + beta * p

		r = r_next
		p = p_next

	return x0