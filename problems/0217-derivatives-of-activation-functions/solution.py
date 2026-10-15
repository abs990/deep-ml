import math

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	exp_x = math.exp(x)
	exp_neg_x = math.exp(-x)

	# sigmoid
	f_sigmoid = 1 / (1 + exp_neg_x)

	# tanh
	f_tanh =  (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

	return {
		'sigmoid': f_sigmoid * (1 - f_sigmoid), 
		'tanh': 1 - (f_tanh ** 2), 
		'relu': 1 if x > 0 else 0
	}