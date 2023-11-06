import numpy as np

class Sigmoid:
    
    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    
    def backward(self, x):
        return (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x))))
    
class Softmax:
    
    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1)
    
class CCELoss:
    
    def __call__(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10**-100))
    
    def backward(self, y_true, y_pred):
        # size? (minibatches,1) or (minibatches,y)?
        return y_pred - y_true