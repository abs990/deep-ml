import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    n_q, d_k = Q.shape

    scores = (Q @ K.T) / np.sqrt(d_k)

    # numerically stable row-wise softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    scores = np.exp(scores)
    scores = scores / np.sum(scores, axis=1, keepdims=True)

    return np.round(scores, 4).tolist()