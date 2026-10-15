import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	d = 1.0

	for fn in reversed(functions):
		if fn == 'square':
			d *= 2 * x
			x = x ** 2
		elif fn == 'sin':
			d *= np.cos(x)
			x = np.sin(x)
		elif fn == 'exp':
			d *= np.exp(x)
			x = np.exp(x)
		else:
			d /= x
			x = np.log(x)
	
	return d