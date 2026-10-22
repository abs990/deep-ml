import numpy as np

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    
    # naive
    scores_exp = np.exp(scores)
    naive = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    if np.any(np.isnan(naive)):
        naive[:] = np.nan

    # stable
    scores -= np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores)
    stable = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # diff
    max_abs_diff = np.max(np.abs(naive - stable))

    return {
        'naive': np.round(naive, 6).tolist(),
        'stable': np.round(stable, 6).tolist(),
        'max_abs_diff': np.round(max_abs_diff, 6).tolist()
    }