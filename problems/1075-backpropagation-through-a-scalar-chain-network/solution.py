import numpy as np

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def d_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))

    z = []
    a = [a0]

    for w_i, b_i in zip(w, b):
        z.append(w_i * a[-1] + b_i)
        a.append(sigmoid(z[-1]))

    N = len(w)

    dW = [0] * N
    dB = [0] * N

    delta = 2 * (a[-1] - y)

    for idx in range(N - 1 , -1, -1):
        grad_z = delta * a[idx + 1] * (1 - a[idx + 1])
        dW[idx] = round(grad_z * a[idx], 6)
        dB[idx] = round(grad_z, 6)
        delta = grad_z * w[idx]

    return {
        'dW': dW, 
        'dB': dB
    }