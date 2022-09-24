import numpy as np

def RMSE(y_pred, y_true):
    """
    Implementation of a function to calculate the root mean squared error

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        The root mean squared error
    """
    return np.sqrt(np.mean((y_pred - y_true)**2))

def MSE(y_pred, y_true):
    """
    Implementation of a function to calculate the mean squared error

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        The mean squared error
    """
    return np.mean((y_pred - y_true)**2)