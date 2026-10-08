import numpy as np

def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    indices = [row.index(max(row)) for row in scores]
    return [candidates[r][c] for r,c in zip(np.arange(0, len(candidates)), indices)]