import numpy as np

def additive_attention(query, keys, W_q, W_k, v):
    """
    query: (d_q,), keys: (n, d_k), W_q: (d_a, d_q), W_k: (d_a, d_k), v: (d_a,)
    Returns (weights (n,), context (d_k,)).
    """
    # Your code here
    e = (np.tanh(W_q @ query + keys @ W_k.T)) @ v
    
    # numerically stable softmax
    e = np.exp(e - np.max(e))
    e = e / np.sum(e)

    return e, keys.T @ e