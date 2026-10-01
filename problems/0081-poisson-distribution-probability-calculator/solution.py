from math import log, exp

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	# Your code here
	log_p = (k * log(lam)) - lam - sum([log(i) for i in range(1, k + 1)])
	return round(exp(log_p), 5)