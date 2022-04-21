import lossfun 
import numpy as np


class Model():
    def __init__(self):
        self.layers = []


    def addLayer(self, layer):
        self.layers.append(layer)


    def forward(self, inputs):
        inputs = np.array(inputs)
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        for layer in self.layers:
            inputs = layer.forward(inputs, train=False)
        return inputs
    
    def backward(self, inputs, targets):
        inputs = np.array(inputs)
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        targets = np.array(targets)
        if targets.shape == 1:
            targets = np.reshape(targets, (1, targets.shape))
        for layer in self.layers:
            inputs = layer.forward(inputs, train=True)
        #Now the input is a matrix of forward pass output
        #Calculate the loss
        loss_function = lossfun.crossEnthropy
        loss = loss_function.forward(inputs, targets)
