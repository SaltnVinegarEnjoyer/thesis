import numpy as np
import math

#The gradients may be stored in the instances of activation functions themselves later on
#I still haven't descided the architecture yet
class Sigmoid:
    def forward(self, x):
        self.x = x
        #This is just an example. I am going to make a more optimized implementation later on.
        x = np.array(x)
        #Remember the original shape
        dims = np.shape(x)
        #Reshape an array to 1D
        x = x.reshape(-1)
  
        #Loop through each of the element and apply expit to it
        for i in range(len(x)):
            x[i] = (1/(1+math.exp(-x[i])))
  
        #Reshape back to original shape
        x = x.reshape(dims)
        return x
    
    def backward(self, nextLoss):
        #Derivative of a sigmoid is result * (1 - result)
        return (self.forward(self.x) * (1 - self.forward(self.x)))

        pass

#max(0, x)
class Relu:
    def forward(self, x):
        x = np.maximum(0, x)
        self.out = x
        return x
    
    def backward(self, next_grad):
        #dReLU/dx = 1 if x > 0, 0 if x <= 0
        res = next_grad.copy()
        #Change each value at index i to 0 when output[i] of a function is < 0
        res = np.where(res < 0, 0, res)
        return res

class Softmax:
    def forward(self, x):
       #Get an exponintiated array
       exponents = np.exp(x)
       #Get the dimensions 
       dims = np.shape(x)
       #Get a vector of sums(as per output set)
       expsum = np.sum(exponents, axis=1)
       #Reshape the output, so that each value is a vector
       expsum = expsum.reshape(dims[0], 1)

       #Divide exponentiated values by the sum of each exponintiated output set(dim 1)
       self.norm = exponents / expsum
       return self.norm

    def backward(self, next_grad):
        #Gradient placeholder
        gradient = np.zeros_like(next_grad)

        #For each grad set in a batch
        for index, (out, grad) in enumerate(zip(self.norm, next_grad)):
            #Flatten the out from next grad
            out = out.reshape(-1, 1)
            #Calculate the Jacobian matrix
            #Diagflat - array which is one-hot encoded by the values at indices(put all the values diagonally)
            jacobian = np.diagflat(out) - np.dot(out, out.T)

            #Append the gradient list
            gradient[index] = np.dot(jacobian, grad)
        
        #Return the gradient for further propagation
        return gradient