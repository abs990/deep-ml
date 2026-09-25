import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = np.sort(np.array(data))

    stats = {}

    stats['mean'] = np.mean(data)
    stats['median'] = np.median(data)

    values, counts = np.unique(data, return_counts=True)
    stats['mode'] = values[np.argmax(counts)]

    stats['variance'] = np.mean(np.square(np.subtract(data, stats['mean'])))
    stats['standard_deviation'] = np.sqrt(stats['variance'])
    stats['25th_percentile'] = np.quantile(data, 0.25)
    stats['50th_percentile'] = stats['median']
    stats['75th_percentile'] = np.quantile(data, 0.75)
    stats['interquartile_range'] = stats['75th_percentile'] - stats['25th_percentile']

    return stats
