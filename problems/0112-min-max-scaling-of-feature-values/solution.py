def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    x_min = min(x)
    x_max = max(x)

    # special case
    if x_min == x_max:
        return [0] * len(x)

    min_max_diff = x_max - x_min
    return [(a - x_min) / min_max_diff for a in x]