import numpy as np
from activation_functions import sigmoid, softmax


class MLP_layer:
    
    def __init__(self, n_inputs, n_units, activation_func):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.weights = np.zeros((n_inputs + 1,n_units))
        self.weights = self.init_weights(0, 0.2)
        self.activation_func = activation_func
        
    def init_weights(self, mean, std):
        self.weights[0:self.n_inputs, :] = random.normal(mean, std, ((n_inputs,n_units)))
        
    def set_weights(self, weights):
        self.weights = weights
        
    def forward(self, x):
        x = np.concatenate(x, np.ones((len(x),1))) #add 1s for bias
        net_input = x @ self.weights
        y = self.activation_func(net_input)
        return Y
        