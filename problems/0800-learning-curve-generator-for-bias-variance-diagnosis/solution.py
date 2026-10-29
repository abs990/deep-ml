import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train, y_train, X_val, y_val = np.array(X_train).flatten(), np.array(y_train), np.array(X_val).flatten(), np.array(y_val)

    result = {
        'train_errors': [], 
        'val_errors': [], 
        'diagnosis': None
    }

    X_val_v = np.vander(X_val, N=degree+1, increasing=True)

    for n in train_sizes:
        X_t = np.vander(X_train[:n], N=degree+1, increasing=True)
        y_t = y_train[:n]
        W = np.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)
        result['train_errors'].append(float(np.mean((X_t @ W - y_t) ** 2)))
        result['val_errors'].append(float(np.mean((X_val_v @ W - y_val) ** 2)))

        if result['train_errors'][-1] > bias_threshold:
            result['diagnosis'] = 'high_bias'
        elif result['val_errors'][-1] - result['train_errors'][-1] > variance_threshold:
            result['diagnosis'] = 'high_variance'
        else:
            result['diagnosis'] = 'good_fit'

    return result