def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	matrix = 1.0 * np.array(matrix)
	means = np.mean(matrix, axis=1 if mode == "row" else 0)
	return means