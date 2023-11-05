import numpy as np

class Sigmoid:
    
    def __call__(self, x):
        return 1/(1 + exp(-x))
    
class Softmax:
    
    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1)
    
class CCELoss:
    
    def __call__(self, y):