import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    T, _ = attn_weights.shape
    if method == 'tril':
        mask = np.tril(np.ones((T, T), dtype=bool), k=0)
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    else:
        mask = np.triu(np.ones((T, T), dtype=bool), k=0)
        np.fill_diagonal(mask, 0)
        
        attn_weights = np.log(attn_weights)
        attn_weights = np.where(mask, -np.inf, attn_weights)

        # stable softmax
        attn_weights = attn_weights - np.max(attn_weights, axis=-1, keepdims=True)
        attn_weights = np.exp(attn_weights)
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    
    return attn_weights.tolist()