import numpy as np

class Dense:

    # Layer initialization
    def __init__(self, inputs, outputs, actfun):
        #1 column - weight set per neuron e.g. TRANSPOSED FROM THE INITIALIZATION. This is just an optimization.
        #We need to have small weights at initialization. This will ease the training process, since the changes made during initial training will make a considerable effect on output
        self.weights = 0.01 * np.random.randn(inputs, outputs)
        #row - 1, since there's 1 bias per neuron, column - amount of neurons
        self.biases = np.zeros((1, outputs))
        #Now the activation function is received as an object
        self.actfun = actfun()

    # Forward pass
    def forward(self, inputs):
        #Memorize input values for the backpropagation
        self.inputs = inputs
        #Ordinary forward operation of neural network
        #Memorize output for the backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases
        #I will not return anything from the forward function
        #Most of the logic is now in the model class
        self.actfun.forward(self.output)

    # Backward pass
    def backward(self, actfun_next_grad):
        #Backpropagate through the activation function first
        self.actfun.backward(actfun_next_grad)

        #The derivative of a neuron: f(wei,inp,bias) = inp * wei + bias
        #df(w,i,b)/di = w. inp is also a function, so f' = w * f'(i,w,b) -> w1 ...
        #This means that di for each neuron is going to be a matrix of weights. [[w1,w2,w3]]
        #To find di, we also need to know dx.

        #self.actfun.input_gradient - gradient obtained from the layer's activation function
        #It is a matrix of derivatives as per neuron output
        #Each row - a vector of derivatives for each training set in a batch

        #Set the weight and bias gradients for the neurons
        #We need to transpose weights so that we are matching the shape of inputs(they are NOT transposed from the initialization)
        self.weight_gradient = np.dot(self.inputs.T, self.actfun.input_gradient)

        #Find bias gradient for optimizing
        #Derivative of a bias calculation is always 1. By the chain rule, we just need to get the overall gradient that we got from the next layer and multiply it by 1.
        #We also need to add another dimension since we have lost one in the sum() function
        self.bias_gradient = np.array([np.sum(self.actfun.input_gradient, axis=0)])

        #Set the input gradient for further backpropagation
        #We need to transpose weights so that we are matching the shape of inputs(they are NOT transposed from the initialization)
        self.input_gradient = np.dot(self.actfun.input_gradient, self.weights.T)