import numpy as np

class Sigmoid:
    
    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    
    def backward(self, x):
        return (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x))))
    
class Softmax:
    
    # not implemented yet
    
    def __call__(self, z):
        result = np.exp(z) / np.sum(np.exp(z), axis=1)
        print(f"input_softmax: {z}\noutput_softmax: {result}")
        return result
    
    def backward(self, batch):
        # not implemented yet
        new_batch = []
        for z in batch:
            new_z = []
            s = self.__call__(z)
            for i , z_i in enumerate(z):
               new_z.append()
                
        return np.array(result)
    
class CCELoss:
    
    def __call__(self, y_true, y_pred):
        y_pred[y_pred == 0.0] = 10**-100
        y_pred[y_pred < 0.0] = np.abs(y_pred[y_pred < 0.0])
        return -np.sum(y_true * np.log(y_pred))
    
    def backward(self, y_true, y_pred):
        # size (minibatchesize,10)
        return y_pred - y_true