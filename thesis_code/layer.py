import numpy as np
from im2col import *

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
        #Initialize the weight momentum for the optimization
        self.weight_momentum = np.zeros_like(self.weights)
        #Initialize the bias momentum for the optimization
        self.bias_momentum = np.zeros_like(self.biases)

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

class Conv:
    def __init__(self, img_dims, filters_amount, filter_size, stride, padding):
        #img_dims is a tuple: (channels, height, width)
        self.input_channels, self.input_height, self.input_width = img_dims
        self.filters_amount = filters_amount
        #We will allways have filters of n*n size, so we don't need to differentiate filter's height and width
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        #Initialize weights matrix
        self.weights = 0.001 * np.random.randn(filters_amount, self.input_channels, filter_size, filter_size)
        #Bias vector (1 bias per filter). We need additional dimension
        self.bias = np.zeros((filters_amount, 1))
        #Formula for finding the output dimensions:
        #((input height - filter size + 2*padding) / stride) +1 x ((input_width - filter_size + 2*padding) / stride)+1
        self.out_width = ((self.input_width - self.filter_size + 2*padding) / stride) + 1
        self.out_height = ((self.input_height - self.filter_size + 2*padding) / stride) + 1


    def forward(self, inputs):
        #Memorize the inputs for backpropagation
        self.inputs = inputs

        #Transform inputs to matrix, where each column is a "convolutional place" of the image
        self.inputs_colonized = im2col_indices(inputs, self.filter_size, self.filter_size, self.padding, self.stride)
        #Transform weights to the matrix, where each column is a single set of filter's weights
        weights_colonized = self.weights.reshape(self.filters_amount, self.input_channels * self.filter_size * self.filter_size)
        #Apply filters(mutiply weights by the corresponding pixel value)
        out = np.dot(weights_colonized, self.inputs_colonized) + self.bias
        #Shape back the resulting matrix to normal image batch form
        self.output = out.reshape(self.filters_amount, self.out_height, self.out_width, self.input_amount).transpose(3, 0, 1, 2)