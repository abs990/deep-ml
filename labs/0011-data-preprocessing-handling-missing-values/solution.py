import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # approach: give estimate as column mean
    column_means = np.nanmean(X_clean, axis=0)
    rows, cols = np.where(np.isnan(X_clean))
    X_clean[rows, cols] = column_means[cols]
    
    return X_clean
