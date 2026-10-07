import math
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	determinant = 0

	N = len(matrix)
	if N == 3:
		# 3x3 case
		determinant += matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) 
		determinant -= matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) 
		determinant += matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
	else:
		# invoke recursively
		determinant = 0
		
		for i in range(4):
			minor = [
				[matrix[r][c] for c in range(4) if c != i] 
				for r in range(1, 4)]
			minor_det = determinant_4x4(minor)
			determinant += matrix[0][i] * (math.pow(-1, i)) * minor_det
		
	return determinant