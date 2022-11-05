class Node:
    def __init__(self, bias, activation_function):
        self.bias = bias
        self.gradient = 0
        self.activation_function = activation_function
    
    def get_activation_result(self):
        return self.activation_function(self.data)
    
    def get_gradient(self):
        return self.gradient

    def reset_gradient(self):
        self.gradient = 0
    
    def get_bias(self):
        return self.bias

class Layer:
    def __init__(self, number_of_nodes, activation_function):
        self.nodes = []
        for _ in range(number_of_nodes):
            self.nodes.append(Node(0, activation_function))
    
    def get_nodes(self):
        return self.nodes
    
