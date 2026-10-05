import numpy as np
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    n, c, h, w = x.shape

    # dimension calculations
    output_height = math.ceil((h - 1)/stride)
    output_width = math.ceil((w - 1)/stride)

    output = np.zeros((n, c, output_height, output_width), dtype=x.dtype)

    for n_idx in range(n):
        for c_idx in range(c):
            for i in range(output_height):
                for j in range(output_width):
                    i_start = i * stride
                    j_start = j * stride
                    i_end = min(h, i_start + kernel_size)
                    j_end = min(w, j_start + kernel_size)

                    output[n_idx, c_idx, i, j] = np.max(x[n_idx, c_idx, i_start:i_end, j_start:j_end])

    return output