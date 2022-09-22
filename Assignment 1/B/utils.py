import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def make_test_set(data, test_size, random_seed=0):
    """
    Implementation of a function to split the data into training and test set

    Args:
        data: Data to split
        test_size: Percentage of data to use as test set
        random_seed: A seed to make the randomization deterministic and reproducible
    
    Returns:
        A tuple containing the training and test set
    """
    np.random.seed(random_seed)
    # np.random.shuffle(data)
    test_size = int(len(data) * test_size)
    return (data[:-test_size], data[-test_size:])


def save_weights(params, bias, save_path, fileName):
    """
    Implementation to Save Model Params and Bias after training

    Args:
        params: Parameters of the Model
        bias: Bias of the Model
        save_path: Path to save the model
        fileName: Name of the file to save the model

    Returns:
        None
    """
    with open(os.path.join(save_path, fileName)) as f:
        np.save(f, params)
        np.save(f, bias)

def load_weights(load_path, fileName):
    """
    Implementation to Load Model Params and Bias which were previously saved

    Args:
        load_path: Path to load the model
        fileName: Name of the file to load the model

    Returns:
        A tuple containing the model parameters and bias
    """
    with open(os.path.join(load_path, fileName)) as f:
        params = np.load(f)
        bias = np.load(f)
    return (params, bias)

def log_message(message, log_file):
    """
    Implementation of a logger to log the training process

    Args:
        message: Message to log
        log_file: File to log the message
    """
    with open(log_file, 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write("\t" + message + '\n')

def save_to_json(data, path):
    """
    Implementation to save the data to a JSON file

    Args:
        data: Data to save
        path: Path to save the data
    """
    j = json.dumps(data, indent=4)
    f = open(path,"w")
    f.write(j)
    f.close()

def EarlyStopping(ls_loss, patience, min_delta):
    """
    Implementation of Early Stopping to automate manual oversight of the training process by stopping the training process

    Args:
        ls_loss: A list containing the loss values over the epochs
        patience: Number of epochs to check for improvement and stop the training process if no improvement is observed
        min_delta: Change in loss value which is considered as an improvement

    Returns:
        A bool value indicating whether to stop the training process or not
    """
    if len(ls_loss) > patience:
        if ls_loss[len(ls_loss)-1] - ls_loss[len(ls_loss)-patience] > min_delta:
            return True
    return False

def min_max_normalization(data):
    """
    Implementation of min-max normalization to scale the features per column. The features are scaled similar to how `sklearn.preprocessing.MinMaxScaler` works

    Args:
        data: Data to Normalize
    
    Returns:
        A numpy array containing the scaled features and the min and max values used for scaling
    """
    return (data - data.amin()) / (data.amax() - data.amin()), data.amin(), data.amax()

def standard_scaler(data):
    """
    Implementation of Standard Scaler to scale the features. The features are scaled similar to how `sklearn.preprocessing.StandardScaler` works

    Args:
        data: Data to Normalize
    
    Returns:
        A numpy array containing the scaled features and the mean and standard deviation used for scaling
    """
    return (data - data.mean()) / data.std(), data.mean(), data.std()