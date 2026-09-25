import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = data.shape[0]

    # shuffle with seed for reproducibility
    rows = np.random.default_rng(seed).permutation(n)
    data = data[rows, :]

    split_data = []

    # training set
    train_end = int(n * train_frac)
    split_data.append(data[:train_end, :])

    # validation set
    validation_end = train_end + int(n * validation_frac)
    split_data.append(data[train_end:validation_end, :])

    # test set
    split_data.append(data[validation_end:, :] if validation_end < n else np.array([]))

    return split_data
