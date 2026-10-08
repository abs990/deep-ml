import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:

    vectors = np.array(vectors)
    m, _ = vectors.shape

    # Gram-Schmidt
    basis = []

    # step 1
    idx = 0
    for idx in range(m):
        first_norm = np.linalg.norm(vectors[idx])
        if first_norm > tol:
            basis.append(vectors[idx] / first_norm)
            break

    # step 2
    for k in range(1 + idx, m):
        dp = np.array(basis) @ vectors[k].T
        w = vectors[k] - np.sum(dp[:, None] * basis, axis=0)
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)

    return basis
