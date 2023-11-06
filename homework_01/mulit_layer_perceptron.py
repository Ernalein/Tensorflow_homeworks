import numpy as np

from perceptron_layer import MLP_layer
from activation_functions import Sigmoid, Softmax, CCELoss


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
    
    def backward(self, loss_func, y_true, y_pred, learning_rate):
        # error_signal size -> (minibatchsize, n_units)
        error_signal = loss_func(y_true, y_pred)
        for layer in reversed(self.layers):
            error_signal = layer.backward(error_signal, learning_rate)
        
    def training(self, data_X, data_Y, epochs, learning_rate=0.001, loss_func=CCELoss()):
        average_loss = []
        for epoch in range(epochs):
            epoch_loss = []
            for x_batch, y_batch in zip(data_X, data_Y):
                outputs = self.forward(x_batch)
                loss = loss_func(y_batch, outputs)
                epoch_loss.append(loss)              
                self.backward(loss_func, y_batch, outputs, learning_rate)
            average_loss.append(np.mean(epoch_loss))
            print(epoch_loss)
            
        # plot average loss
        
    #def ploting_loss(self, average_losses, n_epochs):