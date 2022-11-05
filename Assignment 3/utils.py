import pandas as pd
import numpy as np
import struct

def load_mnist():
    train_image_path = r"MNIST Data\train-images.idx3-ubyte"
    train_label_path = r"MNIST Data\train-labels.idx1-ubyte"
    test_image_path = r"MNIST Data\t10k-images.idx3-ubyte"
    test_label_path = r"MNIST Data\t10k-labels.idx1-ubyte"
    # Load training data
    with open(train_image_path, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows * ncols))
    with open(train_label_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,))
    # Load testing data
    with open(test_image_path, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))
    with open(test_label_path, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,))
    return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)