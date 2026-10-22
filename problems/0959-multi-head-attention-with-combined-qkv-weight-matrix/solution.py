import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    batch, num_tokens, d_in = x.shape
    d_out, _ = W_out.shape
    head_dim = d_out // num_heads
    causal_mask = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)

    # Projection. Shape = (batch, num_tokens, 3 * d_out)
    QKV = x @ W_qkv

    # Split first, then reshape 
    Q, K, V = np.split(QKV, 3, axis=-1)# Shape = (batch, num_tokens, d_out)
    Q = Q.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3) # Shape = (batch, num_heads, num_tokens, head_dim)
    K = K.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, num_tokens, num_heads, head_dim).transpose(0, 2, 1, 3)

    # attention shape - (B, num_heads, num_tokens, num_tokens)
    att = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    # Mask future positions with negative infinity
    att = np.where(causal_mask[None, None, :, :], -np.inf, att)

    # stable softmax
    att = att - np.max(att, axis=-1, keepdims=True)
    att = np.exp(att)
    att = att / np.sum(att, axis=-1, keepdims=True)
    
    # result shape = (B, num_heads, num_tokens, head_dim)
    result = att @ V
    result = result.transpose(0, 2, 1, 3)
    result = result.reshape(batch, num_tokens, d_out)

    return result @ W_out