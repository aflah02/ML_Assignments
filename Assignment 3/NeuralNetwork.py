import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nnv import NNV

class NeuralNetwork:
    def __init__(
        self,
        Input_Layer_Size,
        Number_Of_Hidden_Layers,
        List_of_Nodes_Per_Layer,
        Output_Layer_Size,
        Learning_Rate,
        Activation_Function,
        Weight_Initialization_Function,
        Number_Of_Epochs,
        Batch_Size,
    ):
        self.Input_Layer_Size = Input_Layer_Size
        self.Number_Of_Hidden_Layers = Number_Of_Hidden_Layers
        self.List_of_Nodes_Per_Layer = List_of_Nodes_Per_Layer
        self.Output_Layer_Size = Output_Layer_Size
        self.Learning_Rate = Learning_Rate
        self.Activation_Function = Activation_Function
        self.Weight_Initialization_Function = Weight_Initialization_Function
        self.Number_Of_Epochs = Number_Of_Epochs
        self.Batch_Size = Batch_Size
        self.create_layers()
    
    def predict(self, X):
        input = X
        for i in range(len(self.Layers)):
            input = np.dot(input, self.Layers[i])
            input = self.get_activation_result(input)
        return input

    def buildBatches(self, X, y):
        batches = []
        for i in range(0, len(X), self.Batch_Size):
            batches.append((X[i:i + self.Batch_Size], y[i:i + self.Batch_Size]))
        return batches
    
    def train(self, X, y):
        batches = self.buildBatches(X, y)
        for epoch in range(self.Number_Of_Epochs):
            for batch in batches:
                X, y = batch
                self.feed_forward(X)
                self.back_propagation(X, y)
            if epoch % 100 == 0:
                print("Epoch: ", epoch)
                print("Error: ", self.get_error(X, y))
                print("Accuracy: ", self.get_accuracy(X, y))
                print("")

    def feed_forward(self, X):
        input = X
        output = []
        for i in range(len(self.Layers)):
            input = np.dot(input, self.Layers[i])
            input = self.get_activation_result(input)
            output.append(input)
        return output

    def back_propagation(self, X, y):
        pass
        

    def create_layers(self):
        self.Layers = []
        self.Layers.append(self.initialize_weights((self.Input_Layer_Size, self.List_of_Nodes_Per_Layer[0])))
        for i in range(self.Number_Of_Hidden_Layers - 1):
            self.Layers.append(self.initialize_weights((self.List_of_Nodes_Per_Layer[i], self.List_of_Nodes_Per_Layer[i + 1])))
        self.Layers.append(self.initialize_weights((self.List_of_Nodes_Per_Layer[-1], self.Output_Layer_Size)))

    def initialize_weights(self, shape):
        if self.Weight_Initialization_Function == "gaussian":
            return np.random.normal(0, 1, shape)
        elif self.Weight_Initialization_Function == "zeros":
            return np.zeros(shape)
        elif self.Weight_Initialization_Function == "random":
            return np.random.rand(shape[0], shape[1])

    def get_activation_result(self, x, alpha=0.01):
        if self.Activation_Function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.Activation_Function == "tanh":
            return np.tanh(x)
        elif self.Activation_Function == "relu":
            return np.maximum(0, x)
        elif self.Activation_Function == "leaky_relu":
            return np.where(x > 0, x, x * alpha)
        elif self.Activation_Function == "softmax":
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        elif self.Activation_Function == "linear":
            return x
    
    def get_activation_derivative(self, x, alpha=0.01):
        if self.Activation_Function == "sigmoid":
            return x * (1 - x)
        elif self.Activation_Function == "tanh":
            return 1.0 - x**2
        elif self.Activation_Function == "relu":
            return np.where(x > 0, 1, 0)
        elif self.Activation_Function == "leaky_relu":
            return np.where(x > 0, 1, alpha)
        elif self.Activation_Function == "linear":
            return 1

    def visualize_layers(self):
        plt.rcParams["figure.figsize"] = (200,10)
        layers_list = [
            {"title": f"Input Layer\n{self.Activation_Function}\n{self.Input_Layer_Size}", "units": self.Input_Layer_Size, "color": "red"},
        ]
        for i in range(self.Number_Of_Hidden_Layers):
            layers_list.append({"title": "Hidden Layer " + str(i + 1) + f"\n{self.Activation_Function}\n{self.List_of_Nodes_Per_Layer[i]}", "units": self.List_of_Nodes_Per_Layer[i], "color": "blue"})
        layers_list.append({"title": f"Output Layer\nSoftmax\n{self.Output_Layer_Size}", "units": self.Output_Layer_Size, "color": "green"})
        fig = NNV(layers_list, max_num_nodes_visible=15, node_radius=4, spacing_layer=60, font_size=24)
        fig.render(save_to_file="NeuralNetwork.png")

if __name__ == "__main__":
    nn = NeuralNetwork(28*28, 4, [256,128,64,32], 10, 0.01, "sigmoid", "gaussian", 100, 32)
    nn.visualize_layers()
    


    
    