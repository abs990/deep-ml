import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X = np.array(X)
    
    # attention scores
    att = X @ X.T

    # normalize + softmax
    att = att - np.mean(att, axis=1, keepdims=True)
    att = np.exp(att)
    att = att / np.sum(att, axis=1, keepdims=True)

    return att @ X