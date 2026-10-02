def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	return z * (alpha if z <=0 else 1)
