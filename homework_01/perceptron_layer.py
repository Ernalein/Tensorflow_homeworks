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
        self.stored_input = np.concatenate((x, np.ones((len(x),1))), axis = 1) #add 1s for bias
        self.stored_net_input = self.stored_input @ self.weights
        y = self.activation_func(self.stored_net_input)
        return y
    
    def weights_backward(self, gradients, learning_rate):
        
        self.weights = self.weights + (learning_rate * gradients)
        
    def backward(self, partial_error, learning_rate):
        
        # partial_error size -> (minibatchsize, n_units)
        # stored_net_input size -> (minibatchsize, n_units)
        # error_batches size -> (minibatchsize, n_units)
        error_batches = partial_error * self.activation_func.backward(self.stored_net_input)
        
        # error_batches size -> (minibatchsize, n_units)
        # stored_input size -> (minibatchsize, n_inputs)
        # gradients size -> (n_inputs + 1, n_units) just like the weight matrix
        gradients = np.zeros(self.weights.shape)
        for error, inputs in zip(error_batches, self.stored_input): # iterate through the minibatch
            error = np.expand_dims(error, axis=0)
            inputs = np.expand_dims(inputs, axis=1)
            gradients = gradients + (inputs @ error)
        
        self.weights_backward(gradients, learning_rate)
        
        # error_batches size -> (minibatchsize, n_units)
        # self.weights[0:self.n_inputs] size -> (n_inputs, n_units) -> (without biases)
        # next_partial_error size -> (minibatchesize, n_inputs)
        next_partial_error = (error_batches @ self.weights[0:self.n_inputs].T)
        return next_partial_error