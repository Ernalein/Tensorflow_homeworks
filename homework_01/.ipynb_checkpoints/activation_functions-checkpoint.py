import numpy as np

class Sigmoid:
    
    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    
    def backward(self, x):
        return (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x))))
    
class Softmax:
    
    # not implemented yet
    
    def __call__(self, z):
        exp = np.exp(z)
        return exp / np.expand_dims(np.sum(exp, axis=1), axis=1)
        #print(f"input_softmax: {z}\noutput_softmax: {result}")
        return result
    
    def backward(self, z):
        result_batch = []
        
        # z size (minibatchsize, n_units)
        # s_batch -> (minibatchsize, n_units)
        s_batch = self.__call__(z)
        
        for s in s_batch:
            
            # make the matrix whose size is n_units^2.
            jacobian_m = np.diag(s)

            for i in range(len(jacobian_m)):
                for j in range(len(jacobian_m)):
                    if i == j:
                        jacobian_m[i][j] = s[i] * (1 - s[i])
                    else: 
                        jacobian_m[i][j] = -s[i] * s[j]
                        
            result_batch.append(np.sum(jacobian_m, axis = 0))
        # result size -> (minibatchsize, n_units))
        result_batch = np.array(result_batch)
        
        #print(f"input_softmax_der: {z}\noutput_softmax_der: {result_batch}")
        return result_batch
    
class CCELoss:
    
    def __call__(self, y_true, y_pred):
        y_pred[y_pred == 0.0] = 10**-100
        y_pred[y_pred < 0.0] = np.abs(y_pred[y_pred < 0.0])
        return -np.sum(y_true * np.log(y_pred))
    
    def backward(self, y_true, y_pred):
        # size (minibatchesize,10)
        # return y_pred - y_true
        return - y_true/y_pred