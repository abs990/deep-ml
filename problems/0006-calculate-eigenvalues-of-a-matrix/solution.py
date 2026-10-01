def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	# trace and determinant
	trace = matrix[0][0] + matrix[1][1]
	determinant = matrix[1][1] * matrix[0][0] - matrix[0][1] * matrix[1][0]

	# solve quadratic equation
	from math import sqrt
	d = sqrt((trace * trace) - (4 * determinant))
	eigenvalues = [0.5 * (trace + d), 0.5 * (trace - d)] 

	return sorted(eigenvalues, reverse=True)