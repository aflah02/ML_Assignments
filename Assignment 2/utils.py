import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class dataset:
    def __init__(self, number_of_points):
        self.number_of_points = number_of_points

    def get(self, addNoise=False):
        choices = [0, 1]
        pos_neg = [-1, 1]
        dataset = []
        choices_sampled = np.random.choice(choices, self.number_of_points)
        for choice in choices_sampled:
            pos_or_neg = np.random.choice(pos_neg)
            if choice == 0:
                x_coord = np.random.uniform(-1,1)
                y_coord = pos_or_neg*np.sqrt(1 - x_coord**2)
            if choice == 1:
                x_coord = np.random.uniform(-1,1)
                y_coord = 3+pos_or_neg*np.sqrt(1 - x_coord**2)
            if addNoise:
                x_coord += np.random.normal(0, 0.1)
                y_coord += np.random.normal(0, 0.1)
            dataset.append([x_coord, y_coord, choice])
        return np.array(dataset)

def plot_dataset(dataset, title):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    plt.title(title)
    sns.scatterplot(x=dataset[:,0], y=dataset[:,1], hue=dataset[:,2].astype('int32'), palette="Set2")
    return plt

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100, min_error_threshold=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_error_threshold = min_error_threshold
        self.weights = None
        self.bias = None

    def train(self, dataset, initialization='gaussian', bias_initialization='gaussian', bias=None, verbose=0, fixedBias=False):
        data_size = dataset.shape[-1] - 1

        weights = self._initialize_weights(data_size, initialization)

        if bias is None:
            bias = self._initalize_bias(bias_initialization)

        for epoch in range(self.epochs):

            epoch_error = 0

            for data_point in dataset:

                y_hat = self._activation(self._forward(data_point, weights, bias))

                error = data_point[-1] - y_hat

                epoch_error += error**2

                weights, bias = self._backward(weights, bias, error, data_point, fixedBias)

            if verbose == 1:
                print(f'Epoch: {epoch}, Error: {epoch_error}')

            if self.min_error_threshold is not None:
                if epoch_error < self.min_error_threshold:
                    break
        
        self.weights = weights
        self.bias = bias

        return weights, bias

    def predict(self, data_point):
        return self._activation(self._forward(data_point[:2], self._get_weights(), self._get_bias()))

    def plot_decision_boundary(self, dataset, title):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 10))
        plt.title(title)
        sns.scatterplot(x=dataset[:,0], y=dataset[:,1], hue=dataset[:,2].astype('int32'), palette="Set2")
        x = np.linspace(-1, 1, 100)
        y = -(self._get_weights()[0]*x + self._get_bias())/self._get_weights()[1]
        plt.plot(x, y, color='red')
        return plt

    def _forward(self, dataset, weights, bias):
        return np.dot(dataset[:2], weights) + bias
    
    def _backward(self, weights, bias, error, data_point, fixedBias):
        weights += self.learning_rate * error * data_point[:2]
        if not fixedBias:
            bias += self.learning_rate * error
        return weights, bias

    def _activation(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def _get_average_error(self, dataset, weights, bias):
        return np.mean(np.abs(self._activation(self._forward(dataset, weights, bias) - dataset[:,2])))

    def _initialize_weights(self, data_size, initialization):
        if initialization == 'gaussian':
            return np.random.normal(0, 1, data_size)
        elif initialization == 'zeros':
            return np.zeros(data_size)
        else:
            raise ValueError('Initialization method not supported')

    def _initalize_bias(self, initialization):
        if initialization == 'gaussian':
            return np.random.normal(0, 1)
        elif initialization == 'zeros':
            return 0
        else:
            raise ValueError('Initialization method not supported')

    def _get_weights(self):
        if self.weights is None:
            raise ValueError('Weights not initialized')
        else:
            return self.weights

    def _get_bias(self):
        if self.bias is None:
            raise ValueError('Bias not initialized')
        else:
            return self.bias