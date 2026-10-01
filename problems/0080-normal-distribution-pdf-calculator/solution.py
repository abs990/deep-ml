from math import sqrt, exp, pi

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	val = exp(-0.5 * ((x - mean) ** 2) / (std_dev ** 2)) / (std_dev * sqrt(2 * pi))
	return round(val,5)