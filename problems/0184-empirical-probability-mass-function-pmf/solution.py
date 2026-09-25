def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    num_samples = len(samples)
    if num_samples == 0:
        return []
    import numpy as np
    unique, counts = np.unique(samples, return_counts=True)
    counts = np.divide(1.0 * counts, num_samples)
    return list(zip(unique, counts))