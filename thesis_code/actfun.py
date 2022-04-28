import numpy as np
import math

class Sigmoid:
    def forward(self, inputs):
        #This is just an example. I am going to make a more optimized implementation later on.
        self.inputs = np.array(inputs)
        #Remember the original shape
        dims = np.shape(inputs)
        #Reshape an array to 1D
        inputs = inputs.reshape(-1)
  
        #Loop through each of the element and apply expit to it
        for i in range(len(inputs)):
            inputs[i] = (1/(1+math.exp(-inputs[i])))
  
        #Reshape back to original shape
        self.output = inputs.reshape(dims)
    
    def backward(self, next_grad):
        #Derivative of a sigmoid is result * (1 - result)
        self.input_gradient = (self.forward(self.inputs) * (1 - self.forward(self.inputs)))

#max(0, x)
class ReLu:
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        #We are not returning anything now. THe connection logic is held at the layer and model levels

    def backward(self, next_grad):
        #dReLU/dx = 1 if x > 0, 0 if x <= 0
        self.input_gradient = next_grad.copy()

        #Change each value at index i to 0 when actfun's inputs[i] is < 0
        #np.where() somehow doesen't work for this
        self.input_gradient[self.inputs <= 0] = 0

class Softmax:

    def forward(self, inputs):

       #Memorize inputs for the backpropagation
       self.inputs = inputs
       #Get an exponintiated array
       exponents = np.exp(inputs)
       #Get the dimensions 
       dims = np.shape(inputs)
       #Get a vector of sums(as per output set)
       expsum = np.sum(exponents, axis=1)
       #Reshape the output, so that each value is a vector
       expsum = expsum.reshape(dims[0], 1)

       #Divide exponentiated values by the sum of each exponintiated output set(dim 1)
       self.output = exponents / expsum

    def backward(self, next_grad):

        #Gradient placeholder
        self.input_gradient = np.zeros_like(next_grad)

        #For each grad set in a batch
        for index, (out, grad) in enumerate(zip(self.output, next_grad)):
            #Flatten the output from next grad
            out = out.reshape(-1, 1)

            #Calculate the Jacobian matrix
            #Diagflat - matrix which is one-hot encoded by the values at indices(put all the values diagonally)
            jacobian = np.diagflat(out) - np.dot(out, out.T)

            #Append the gradient list
            self.input_gradient[index] = np.dot(jacobian, grad)