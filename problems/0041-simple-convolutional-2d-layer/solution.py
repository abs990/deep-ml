import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# dimension calculations
	double_padding = 2 * padding
	padded_height = input_height + double_padding
	padded_width = input_width + double_padding
	output_height = int(1 + (padded_height - kernel_height) / stride)
	output_width = int(1 + (padded_width - kernel_width) / stride)

	# pad the input
	padded_input = np.zeros((padded_height, padded_width))
	padded_input[padding:(padding + input_height), padding:(padding + input_width)] = input_matrix

	# output 
	output_matrix = np.zeros((output_height, output_width))

	# convolution
	for i in range(0, output_height):
		for j in range(0, output_width):
			pad_i = i * stride
			pad_j = j * stride
			output_matrix[i, j] = np.sum(kernel * padded_input[pad_i:(pad_i + kernel_height), pad_j:(pad_j+kernel_width)])
	
	return output_matrix
