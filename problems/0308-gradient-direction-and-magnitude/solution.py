import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	result = {}
	result['magnitude'] = np.linalg.norm(np.array(gradient))
	if result['magnitude'] != 0:
		result['direction'] = [x / result['magnitude'] for x in gradient]
	else:
		result['direction'] = gradient
	result['descent_direction'] = [-1.0 * x for x in result['direction']]
	return result