import numpy as np

class dense:
    def __init__(self, inputs, outputs, actfun):
        #1 column - weight set per neuron e.g. TRANSPOSED FROM THE INITIALIZATION. This is just an optimization.
        #We need to have small weights at initialization. This will ease the training process, since the changes made during initial training will make a considerable effect on output
        self.weights = 0.01 * np.random.randn(inputs, outputs)
        #row - 1, since there's 1 bias per neuron, column - amount of neurons
        self.biases = np.zeros((1, outputs))
        #Initialize activation function
        actfunObj = actfun()
        self.actfun = actfunObj #lambda x: actfunObj.forward(x)

    def forward(self, inputs, train=False):
        if train:
            #Save the input values for the backward pass
            self.inputs = inputs

        #Ordinary forward operation of neural network
        outputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.actfun.forward(outputs)
        return outputs
    
    def backward(self, inputs, target):
        
        pass

    def innerBackward(self, next_grad):
        #The derivative of a neuron: f(wei,inp,bias) = inp * wei + bias
        #df(w,i,b)/di = w. inp is also a function, so f' = w * f'(i,w,b) -> w1 ...
        #This means that di for each neuron is going to be a matrix of weights. [[w1,w2,w3]]
        #To find di, we also need to know dx.


        #next_grad - gradient obtained from the next layer
        #It is a matrix of derivatives as per neuron output
        #Each row - a vector of derivatives for each training set in a batch

        #First, we need to find gradient that is propagated from activation function
        self.actfun_gradient = self.actfun.backward(next_grad)

        #Find input gradient for worther backpropagation

        #We need to transpose weights so that we are matching the shape of inputs(they are NOT transposed from the initialization)
        self.input_gradient = np.dot(self.actfun_gradient, self.weights.T)

        #Find weight gradient for optimizing
        #We need to transpose inputs so that we are matching the shape of weights(they are transposed from the initialization)
        #Inputs is the first argument to dot because now it is transposed, and dot multiplies rows by cols
        self.weight_gradient = np.dot(self.inputs.T, self.actfun_gradient)

        #Find bias gradient for optimizing
        #Derivative of a bias calculation is always 1. By the chain rule, we just need to get the overall gradient that we got from the next layer and multiply it by 1.
        #We also need to add another dimension since we have lost one in the sum() function
        self.bias_gradient = np.array([np.sum(next_grad, axis=0)])

        #Return the input gradient for the previous layer
        return np.array(self.input_gradient)

class conv:
    pass


