import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    # threshold check
    accepted_indices = [i for i, value in enumerate(scores) if value > threshold]
    accepted_indices = sorted(accepted_indices, key=lambda i: scores[i], reverse=True)

    acceptance_rate = len(accepted_indices) / len(scores)
    
    # rank
    accepted_scores = [scores[idx] for idx in accepted_indices]

    if n_select is not None:
        n_select = min(n_select, len(scores))
        accepted_scores = accepted_scores[:n_select]
        accepted_indices = accepted_indices[:n_select]

    return {
        'accepted_indices': accepted_indices,
        'acceptance_rate': round(acceptance_rate, 4),
        'mean_quality': round(np.mean(accepted_scores), 4) if len(accepted_scores) > 0 else 0.0
    }