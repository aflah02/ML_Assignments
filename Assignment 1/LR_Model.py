from metrics import MSE, RMSE
import numpy as np
from utils import EarlyStopping

class LinearRegression:
    """
    Implementation of Linear Regression
    
    Args:
        learning_rate (float): Learning rate for the model
        epochs (int): Number of epochs to train the model
        early_stopping (bool): Whether to use early stopping or not
        early_stopping_patience (int): Number of epochs to wait before stopping
        early_stopping_delta (float): Minimum change in loss to qualify as an improvement
        regularization (str): Type of regularization to use. Can be None, "Ridge" or "Lasso"
        reg_lambda (float): Regularization parameter
        param_initialization (str): Method to initialize parameters. Can be "gaussian", "zeros" or "ones"
        verbose (bool): Whether to print the loss after every epoch or not
        print_after (int): Number of epochs to wait before printing the loss
        seed (int): Seed for random number generator
    """
    def __init__(self, 
                learning_rate, 
                epochs, 
                early_stopping=False,
                early_stopping_patience=5,
                early_stopping_delta=0.01,
                regularization=None, 
                reg_lambda=0.01, 
                param_initialization="gaussian", 
                verbose=False, 
                print_after=100,
                seed=42):

        self.lr = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.params = None
        self.param_initialization = param_initialization
        self.dataset_size = None
        self.verbose = verbose
        self.seed = seed
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.print_after = print_after
        self.trainRMSEs = []
        self.valRMSEs = []

    def get_params(self):
        return self.params

    def fit(self, X, y, X_val=None, y_val=None):
        # Create Parameters and initialize them
        self.params = self._initialize_params(X.shape[1])
        self.dataset_size = X.shape[0]
        ls_loss = []
        for epoch in range(self.epochs):
            # Forward Pass
            y_hat = self._forward(X)
            # Compute Loss
            loss = self._loss(y, y_hat)

            ls_loss.append(loss)

            self.trainRMSEs.append(RMSE(y, y_hat))

            if X_val is not None and y_val is not None:
                y_hat_val = self._forward(X_val)
                self.valRMSEs.append(RMSE(y_val, y_hat_val))

            if self.early_stopping:
                if EarlyStopping(ls_loss, self.early_stopping_patience, self.early_stopping_delta):
                    break

            self._backward(X, y, y_hat)
            if self.verbose:
                if epoch % self.print_after == 0:
                    print(f"Epoch: {epoch}, Loss: {loss}")

    def _forward(self, X):
        # Simple forward pass to compute prediction
        return np.dot(X, self.params)

    def _backward(self, X, y, y_hat):
        # Backward pass to update parameters
        if self.regularization == "Ridge":
            self.params -= self.lr * (np.dot(X.T, y_hat - y)/self.dataset_size + 2 * self.reg_lambda * self.params)
        elif self.regularization == "Lasso":
            self.params -= self.lr * (np.dot(X.T, y_hat - y)/self.dataset_size + self.reg_lambda * np.sign(self.params))
        else:
            self.params -= self.lr * np.dot(X.T, y_hat - y)/self.dataset_size
    
    def _loss(self, y, y_hat):
        if self.regularization == "Ridge":
            return MSE(y, y_hat) + self.reg_lambda * np.sum(self.params**2)
        elif self.regularization == "Lasso":
            return MSE(y, y_hat) + self.reg_lambda * np.sum(np.abs(self.params))
        else:
            return MSE(y, y_hat)

    def _initialize_params(self, n_features):
        # Initialize parameters based on the initialization method
       if self.param_initialization == "gaussian":
           np.random.seed(self.seed)
           return np.random.randn(n_features)
       elif self.param_initialization == "zeros":
           return np.zeros(n_features)
       elif self.param_initialization == "ones":
           return np.ones(n_features)
       else:
           raise Exception("Invalid parameter initialization method")

    def predict(self, X):
        return self._forward(X)

    def get_trainRMSEs(self):
        return self.trainRMSEs

    def get_valRMSEs(self):
        return self.valRMSEs

    