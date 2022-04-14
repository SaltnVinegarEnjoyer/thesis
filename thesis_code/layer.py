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

    def forward(self, inputs):
        #Ordinary forward operation of neural network
        outputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.actfun(outputs)
        return outputs
    
    def backward(self, inputs, target):
        pass

class conv:
    pass


