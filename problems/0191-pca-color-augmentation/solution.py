import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    
    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components
    
    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    
    image = image.astype(np.float64) # Work in floating point to avoid uint8 arithmetic issues.
    H, W, _ = image.shape # channels given as 3

    # Step 1 - reshape
    image_flat = image.reshape(-1, 3)

    # Step 2 - mean centering
    image_flat = image_flat - image_flat.mean(axis=0, keepdims=True)

    # Step 3 - Compute covariance matrix
    image_cov = np.cov(image_flat, rowvar=False)

    # Step 4 - Eigen vectors and values
    eigenvalues, eigenvectors = np.linalg.eigh(image_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Step 5 - Compute distortion
    eigenvalues = np.clip(eigenvalues, 0.0, None) # Protect against tiny negative values from numerical precision.
    distortion = eigenvectors @ (alpha * np.sqrt(eigenvalues))

    # Step 6 - Apply augmentation
    return np.clip((image + distortion), 0.0, 255.0)