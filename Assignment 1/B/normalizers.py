import numpy as np

class MinMaxNormalizer:
    def __init__(self):
        self.min_value = None
        self.max_value = None

    def fit(self, data):
        self.min_values = np.amin(data, axis=0)
        self.max_values = np.amax(data, axis=0)
    
    def transform(self, data):
        if self.min_values is None or self.max_values is None:
            raise Exception("You need to fit the data first")
        return (data - self.min_values) / (self.max_values - self.min_values)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def transform(self, data):
        if self.mean is None or self.std is None:
            raise Exception("You need to fit the data first")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
