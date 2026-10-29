import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    # Your code here
    out = sigmoid(W @ a_prev + b)
    
    dC_dz = 2 * (out - y) * out * (1 - out)

    return {
        'dW': np.outer(dC_dz, a_prev), 
        'db': dC_dz,  
        'da_prev': W.T @ dC_dz
    }
