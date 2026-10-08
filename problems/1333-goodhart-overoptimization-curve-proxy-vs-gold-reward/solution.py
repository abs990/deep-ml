import math
import numpy as np

def goodhart_curve(proxy, gold, fractions):
    """Trace a proxy-vs-gold overoptimization curve."""
    n = len(proxy)

    mean_proxy = []
    mean_gold = []

    for f in fractions:
        k = max(1, math.ceil(n * f))
        
        # candidate selection
        indexed = list(enumerate(proxy))
        indexed.sort(key=lambda pair: (-pair[1], pair[0]))
        candidates_proxy = np.array([value for _, value in indexed[:k]])
        candidates_gold = np.array([gold[idx] for idx, _ in  indexed[:k]])

        mean_proxy.append(np.mean(candidates_proxy))
        mean_gold.append(np.mean(candidates_gold))

    return {
        'mean_proxy': np.round(mean_proxy, 4).tolist(), 
        'mean_gold': np.round(mean_gold, 4).tolist(),
        'peak_fraction': fractions[np.argmax(mean_gold)],
        'overopt_gap': round(float(np.max(mean_gold) - mean_gold[np.argmin(fractions)]), 4)
        }