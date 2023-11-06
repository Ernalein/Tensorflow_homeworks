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
    
    def backward(self, Loss_func, y_true, y_pred, learning_rate):
        # error_signal size -> (minibatchsize, n_units)
        error_signal = Loss_func(y_true, y_pred)
        for layer in reversed(self.layers):
            error_signal = layer.backward(error_signal, learning_rate)
        
    #def training():
