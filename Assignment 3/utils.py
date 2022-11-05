import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt

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
        _, size = struct.unpack(">II", f.read(8))
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

def plot_curves(mlp):
    # Train and Validation Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(mlp.loss_curve_, 'r', linewidth=3.0)
    plt.legend(['Training Loss'], fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Train and Validation Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(mlp.validation_scores_, 'b', linewidth=3.0)
    plt.legend(['Validation Accuracy'], fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)