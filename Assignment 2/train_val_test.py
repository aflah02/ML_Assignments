import numpy as np

def train_val_test_split(data, seed=10, shuffle=False, train_size=0.7, val_size=0.15):
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(data)
    train, val, test = data[:int(train_size*len(data))], data[int(train_size*len(data)):int((train_size+val_size)*len(data))], data[int((train_size+val_size)*len(data)):]
    return train, val, test