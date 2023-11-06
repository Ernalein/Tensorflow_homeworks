import numpy as np
import random
from activation_functions import Sigmoid, Softmax


class MLP_layer:
    
    def __init__(self, n_inputs, n_units, activation_func):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.weights = np.zeros((n_inputs + 1,n_units))
        self.init_weights(0, 0.2)
        self.activation_func = activation_func
        self.stored_input = None
        self.stored_net_input = None
        
    def init_weights(self, mean, std):
        self.weights[0:self.n_inputs, :] = np.random.normal(mean, std, ((self.n_inputs,self.n_units)))
        
    def set_weights(self, weights):
        self.weights = weights
        
    def forward(self, x):
        self.stored_input = x # store for backpropagaiton
        x = np.concatenate((x, np.ones((len(x),1))), axis = 1) #add 1s for bias
        self.stored_net_input = x @ self.weights
        y = self.activation_func(self.stored_net_input)
        return y
    
    def weights_backward(self, gradients, learning_rate):
        
        # update weights corresponding to gradients and learning rate
        
    def backward(self, partial_error):
        
        # partial_error size -> (minibatchsize, n_units)
        # error size -> (minibatchsize, n_units)
        error = partial_error * self.activation_func.backward(self.stored_net_input)
        
        # gradients size -> (n_inputs, n_units))
        # gradients = error * activationfunction_deriv * stored_input
        weight_gradients = error * self.stored_input
        #self.weights_backward(weight_gradients, learning_rate)
        
        # next partial error size -> (n_inputs)
        # here i should supstract the biases from the weights first
        next_partial_error = (self.weights @ error)
        return next_partial_error