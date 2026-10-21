import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	final_hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)

	for input in input_sequence:
		input = np.array(input)
		final_hidden_state = np.tanh(
			Wx @ input +
			Wh @ final_hidden_state +
			b
		)

	return final_hidden_state.tolist()