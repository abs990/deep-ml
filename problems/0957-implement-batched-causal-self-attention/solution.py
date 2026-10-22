import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    # Your code here
    X, W_query, W_key, W_value = np.array(X), np.array(W_query), np.array(W_key), np.array(W_value)
    B, T, d_in = X.shape
    _, d_out = W_query.shape
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    # Projection: Q, K, V each have shape (B, T, d_out)
    Q, K, V = np.einsum(
        'bsd,kdo->kbso',
        X,
        np.stack([W_query, W_key, W_value])
    )

    # Attention scores: (B, T, T)
    att = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_out)
    att = np.where(causal_mask, -np.inf, att)

    # stable softmax
    att = att - np.max(att, axis=-1, keepdims=True)
    att = np.exp(att)
    att = att / np.sum(att, axis=-1, keepdims=True)

    return (att @ V).tolist()
