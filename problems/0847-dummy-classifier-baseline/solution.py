import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    if strategy == 'constant':
        return [constant] * n_test
    else:
        # generate classes and counts
        from math import floor
        from collections import Counter
        counts = dict(sorted(Counter(y_train).items()))
        if strategy == 'most_frequent':
            return [max(counts, key=lambda k: (counts[k], -k))] * n_test
        elif strategy == 'uniform':
            classes = list(counts.keys())
            num_classes = len(classes)
            return [classes[i % num_classes] for i in range(n_test)]
        else:
            # stratified
            # pass 1 - assign count using empirical frequency
            # pass 2 - allocate remaining preds starting with smallest class
            from itertools import cycle
            total_available_allocation = n_test
            multiplier = 1.0 * n_test / len(y_train)
            counts_fc = {key: value * multiplier for key, value in counts.items()}
            # pass 1
            for (key1, _), (key2, value2) in zip(counts.items(), counts_fc.items()):
                counts[key1] = floor(value2)
                total_available_allocation -= counts[key1]

            # pass 2
            if total_available_allocation > 0:
                counts_fract_sorted_dict = dict(sorted(counts_fc.items(),  key=lambda x: (-(x[1] % 1), x[0])))
                for key, value in cycle(counts_fract_sorted_dict.items()):
                    counts[key] += 1
                    total_available_allocation -= 1
                    if total_available_allocation == 0:
                        break
            return [key for key, value in counts.items() for _ in range(value)]