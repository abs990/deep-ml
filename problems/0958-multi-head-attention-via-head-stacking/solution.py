import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    x = np.array(x)

    # True above the diagonal (future positions)
    _, T, _ = x.shape
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    results = []

    for W_q, W_k, W_v in heads:
        W_q = np.array(W_q)
        W_k = np.array(W_k)
        W_v = np.array(W_v)

        _, d_head = W_q.shape

        # shape - (B, T, d_head)
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v

        # attention shape - (B, T, T)
        att = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
        # Mask future positions with negative infinity
        att = np.where(causal_mask[None, :, :], -np.inf, att)

        # stable softmax
        att = att - np.max(att, axis=-1, keepdims=True)
        att = np.exp(att)
        att = att / np.sum(att, axis=-1, keepdims=True)

        # Shape: (B, T, d_head)
        results.append(att @ V)

    return np.concatenate(results, axis=-1)