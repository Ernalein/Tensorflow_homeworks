import numpy as np


class Sigmoid:
    """
    Implements the Sigmoid activation function.
    """
    def __init__(self):
        pass
    def __call__(self, x):
        """
        Applies the sigmoid activation function to a vector of pre-activations (x).
        """
        return 1/(1 + np.exp(-x))
    def backward(self, x):
        """
        Returns the derivate of the Sigmoid function.
        """
        sigmoid = self.__call__(x)
        return (sigmoid * (1 - sigmoid))
    
class Softmax:
    """
    Implements the Softmax activation function.
    """
    def __init__(self):
        pass
    def __call__(self, z):
        """
        Applies the Softmax function to a vector of layer pre-activations (z).
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    def backward(self, z):
        """
        Calculates the derivative of the Softmax function w.r.t. the vector z.
        """
        # Extract dimensions of z
        batch_size = z.shape[0]
        n = z.shape[1]
        # Create an array of identity matrices
        eye = np.dstack([np.eye(n)] * batch_size)
        # Calculate and combine the derivatives for every element in a batch
        return np.stack([(eye - z.T).T[i] * z[i] for i in range(batch_size)])
    
class CCELoss:
    """
    Implements the categorical cross-entropy loss function.
    """
    def __init__(self):
        pass
    def __call__(self, target, y_hat):
        """
        Calculates the CCE loss for a given prediction and target.
        Arguments:
        target -- The intended target for an image input
        y_hat -- The classification prediction made by the MLP
        """
        return -np.sum(target*np.log(y_hat), axis=-1)
    def backward(self, target, y_hat):
        """
        Returns a simplified derivate of the CCE loss. Arguments represent the same as above.
        """
        return -target/y_hat