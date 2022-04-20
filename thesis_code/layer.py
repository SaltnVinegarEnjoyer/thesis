import numpy as np

class dense:
    def __init__(self, inputs, outputs, actfun):
        #1 column - weight set per neuron e.g. TRANSPOSED FROM THE INITIALIZATION. This is just an optimization.
        #We need to have small weights at initialization. This will ease the training process, since the changes made during initial training will make a considerable effect on output
        self.weights = 0.01 * np.random.randn(inputs, outputs)
        #row - 1, since there's 1 bias per neuron, column - amount of neurons
        self.biases = np.zeros((1, outputs))
        #Initialize activation function
        self.actfun = lambda x: actfun(x)

    def forward(self, inputs, train=False):
        if train:
            #Save the input values for the backward pass
            self.inputs = inputs

        #Ordinary forward operation of neural network
        outputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.actfun(outputs)
        return outputs
    
    def backward(self, inputs, target):
        
        pass

    def innerBackward(self, inputs, next_grad):
        #The derivative of a neuron: f(wei,inp,bias) = inp * wei + bias
        #df(w,i,b)/di = w. inp is also a function, so f' = w * f'(i,w,b) -> w1 ...
        #This means that di for each neuron is going to be a matrix of weights. [[w1,w2,w3]]
        #To find di, we also need to know dx.


        #nextGrad - gradient obtained from the next layer
        #It is a matrix of derivatives as per neuron output
        #Each row - a vector of derivatives for each training set in a batch

        #di
        self.input_gradient = []
        #dw
        self.weight_gradient = []
        #db
        self.bias_gradient = []

        for tset in next_grad:
            #We need to transpose weights again, because they were transposed at the initialization
            gradset = np.dot(tset, self.weights.T)
            self.input_gradient.append(gradset)
        
        return self.input_gradient
        pass

class conv:
    pass


