from perceptron_layer import MLP_layer
from activation_functions import Sigmoid, Softmax


class MLP:
    
    def __init__(self, n_layers, n_units_per_layer, input_size):
        self.layers = []
        input_size = input_size
        for l, n_units in zip(range(n_layers), n_units_per_layer):
            if l + 1 == n_layers:
                # last layer
                self.layers.append(MLP_layer(input_size, n_units, Softmax()))
            else:
                self.layers.append(MLP_layer(input_size, n_units, Sigmoid()))
                input_size = n_units
                
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y
    
    #def backward()
        # calculate the first error signal (CCELoss backward)
        # feed error signal to last layer
            # claculate cradient for weights (and change them)
            # calculate error signal vector for next layer
        # feed error signal to next layer