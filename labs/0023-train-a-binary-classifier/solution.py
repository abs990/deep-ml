import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    ## Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    best_model = None
    best_score = -1
    best_C = None

    # Test different regularization strengths
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        score = accuracy_score(y_val, val_pred)

        if score > best_score:
            best_score = score
            best_C = C
            best_model = model

    return best_model.predict
