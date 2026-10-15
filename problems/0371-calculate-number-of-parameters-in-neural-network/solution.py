def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	param_count = 0

	for layer in layers:
		layer_type = layer['type']
		bias = layer['bias'] if 'bias' in layer else True

		if layer_type == 'dense':
			input_size = layer['input_size']
			output_size = layer['output_size']
			param_count += input_size * output_size + (output_size if bias else 0)
		else:
			in_channels = layer['in_channels']
			out_channels = layer['out_channels']
			kernel_size = layer['kernel_size']
			param_count += in_channels * out_channels * (kernel_size ** 2) + (out_channels if bias else 0)

	return param_count