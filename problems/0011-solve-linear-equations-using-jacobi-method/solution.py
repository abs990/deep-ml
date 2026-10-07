import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	
	x = np.zeros_like((b))
	
	# mask
	mask = np.ones_like(A)
	np.fill_diagonal(mask, 0)

	A_mask = A * mask
	A_diag = np.diagonal(A)

	for _ in range(n):
		x = (b - A_mask @ x) / A_diag

	return np.round(x, 4)