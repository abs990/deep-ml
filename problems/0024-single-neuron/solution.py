import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# layer
	z = []
	for feat in features:
		z.append(
			sum([w * f for w, f in zip(weights, feat)]) + 
			bias
		)

	# activation
	probabilities = [1.0 / (1 + math.exp(-z_i)) for z_i in z]

	# error
	N = len(labels)
	mse = sum([(p - t) ** 2 for p, t in zip(probabilities, labels)]) / N

	# round
	probabilities = [round(p, 4) for p in probabilities]
	mse = round(mse, 4)

	return probabilities, mse