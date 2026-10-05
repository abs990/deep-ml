import numpy as np
import math

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """
    
    n_x, c, h, w = x.shape
    half_n_start = math.floor(n / 2) #- (1 if n % 2 == 0 else 0)
    half_n_end = math.ceil(n / 2) - (1 if n % 2 == 1 else 0)
    c_end_idx = c - 1

    b = np.zeros_like(x)

    for n_idx in range(n_x):
        for c_idx in range(c):
            neighbour_c_idx = np.arange(max(0, c_idx - half_n_start), 1 + min(c_end_idx, c_idx + half_n_end))
            neighbour_sum = np.sum(x[n_idx, neighbour_c_idx, :, :] ** 2, axis=0)

            b[n_idx, c_idx, :, :] = x[n_idx, c_idx, :, :] / ((k + alpha * neighbour_sum) ** beta)

    return b