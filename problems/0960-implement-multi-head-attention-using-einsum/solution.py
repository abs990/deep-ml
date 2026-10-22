import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.

    Args:
        X: input array of shape (B, S, d_in)
        W_q, W_k, W_v: projection matrices of shape (d_in, d_out)
        W_o: output projection matrix of shape (d_out, d_out)
        num_heads: number of attention heads (must divide d_out)
        causal: if True, apply a causal mask so position t only attends to <= t

    Returns:
        Nested list of shape (B, S, d_out) representing the attention output.
    """
    # Your code here
    X, W_q, W_k, W_v, W_o = np.array(X), np.array(W_q), np.array(W_k), np.array(W_v), np.array(W_o)
    B, S, d_in = X.shape
    _, d_out = W_q.shape
    head_dim = d_out // num_heads
    causal_mask = np.triu(np.ones((S, S), dtype=bool), k=1) # only when requested

    # Projection
    Q, K, V = np.einsum('bsd, kdo->kbso', X, np.stack([W_q, W_k, W_v]))

    # Reshape to (B, S, num_heads, head_dim)
    Q = Q.reshape(B, S, num_heads, head_dim)
    K = K.reshape(B, S, num_heads, head_dim)
    V = V.reshape(B, S, num_heads, head_dim)

    # attention shape - (B, num_heads, S, S)
    att = np.einsum('bsnd, btnd->bnst', Q, K) / np.sqrt(head_dim)
    # When requested, mask future positions with negative infinity
    if causal:
        att = np.where(causal_mask[None, None, :, :], -np.inf, att)

    # stable softmax
    att = att - np.max(att, axis=-1, keepdims=True)
    att = np.exp(att)
    att = att / np.sum(att, axis=-1, keepdims=True)

    # result shape = (B, S, num_heads, head_dim)
    result = np.einsum('bnst, btnd->bsnd', att, V)
    result = result.reshape(B, S, d_out)

    return result @ W_o