import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        Implementation of a function to fit the data
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def transform(self, data):
        """
        Implementation of a function to transform the data
        """
        if self.mean is None or self.std is None:
            raise Exception("You need to fit the data first")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        """
        Implementation of a function to fit and transform the data
        """
        self.fit(data)
        return self.transform(data)
