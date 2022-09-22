# kfold crpss validation

import numpy as np
import pandas as pd

# from scratch

def four_fold_split(X, y):
    """
    Implementation of a function to split the data into four folds

    Args:
        X: Data to split
        y: Labels to split

    Returns:
        A tuple containing the four folds
    """
    X1 = X[:int(len(X)/4)]
    X2 = X[int(len(X)/4):int(len(X)/2)]
    X3 = X[int(len(X)/2):int(3*len(X)/4)]
    X4 = X[int(3*len(X)/4):]
    y1 = y[:int(len(y)/4)]
    y2 = y[int(len(y)/4):int(len(y)/2)]
    y3 = y[int(len(y)/2):int(3*len(y)/4)]
    y4 = y[int(3*len(y)/4):]
    # Withhold one and make pairs
    config1 = (np.concatenate((X2, X3, X4), axis=0), np.concatenate((y2, y3, y4), axis=0), X1, y1)
    config2 = (np.concatenate((X1, X3, X4), axis=0), np.concatenate((y1, y3, y4), axis=0), X2, y2)
    config3 = (np.concatenate((X1, X2, X4), axis=0), np.concatenate((y1, y2, y4), axis=0), X3, y3)
    config4 = (np.concatenate((X1, X2, X3), axis=0), np.concatenate((y1, y2, y3), axis=0), X4, y4)
    return (config1, config2, config3, config4)

def two_fold_split(X, y):
    """
    Implementation of a function to split the data into two folds

    Args:
        X: Data to split
        y: Labels to split

    Returns:
        A tuple containing the two folds
    """
    X1 = X[:int(len(X)/2)]
    X2 = X[int(len(X)/2):]
    y1 = y[:int(len(y)/2)]
    y2 = y[int(len(y)/2):]
    # Withhold one and make pairs
    config1 = (X2, y2, X1, y1)
    config2 = (X1, y1, X2, y2)
    return (config1, config2)

def three_fold_split(X, y):
    """
    Implementation of a function to split the data into three folds

    Args:
        X: Data to split
        y: Labels to split

    Returns:
        A tuple containing the four folds
    """
    X1 = X[:int(len(X)/3)]
    X2 = X[int(len(X)/3):2*int(len(X)/3)]
    X3 = X[2*int(len(X)/3):]
    y1 = y[:int(len(y)/3)]
    y2 = y[int(len(y)/3):2*int(len(y)/3)]
    y3 = y[2*int(len(X)/3):]
    # Withhold one and make pairs
    config1 = (np.concatenate((X2, X3), axis=0), np.concatenate((y2, y3), axis=0), X1, y1)
    config2 = (np.concatenate((X1, X3), axis=0), np.concatenate((y1, y3), axis=0), X2, y2)
    config3 = (np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0), X3, y3)
    return (config1, config2, config3)

def five_fold_split(X, y):
    """
    Implementation of a function to split the data into five folds

    Args:
        X: Data to split
        y: Labels to split

    Returns:
        A tuple containing the four folds
    """
    X1 = X[:int(len(X)/5)]
    X2 = X[int(len(X)/5):2*int(len(X)/5)]
    X3 = X[2*int(len(X)/5):3*int(len(X)/5)]
    X4 = X[3*int(len(X)/5):4*int(len(X)/5)]
    X5 = X[4*int(len(X)/5):]
    y1 = y[:int(len(y)/5)]
    y2 = y[int(len(y)/5):2*int(len(y)/5)]
    y3 = y[2*int(len(y)/5):3*int(len(y)/5)]
    y4 = y[3*int(len(y)/5):4*int(len(y)/5)]
    y5 = y[4*int(len(y)/5):]
    # Withhold one and make pairs
    config1 = (np.concatenate((X2, X3, X4, X5), axis=0), np.concatenate((y2, y3, y4, y5), axis=0), X1, y1)
    config2 = (np.concatenate((X1, X3, X4, X5), axis=0), np.concatenate((y1, y3, y4, y5), axis=0), X2, y2)
    config3 = (np.concatenate((X1, X2, X4, X5), axis=0), np.concatenate((y1, y2, y4, y5), axis=0), X3, y3)
    config4 = (np.concatenate((X1, X2, X3, X5), axis=0), np.concatenate((y1, y2, y3, y5), axis=0), X4, y4)
    config5 = (np.concatenate((X1, X2, X3, X4), axis=0), np.concatenate((y1, y2, y3, y4), axis=0), X5, y5)
    return (config1, config2, config3, config4, config5)