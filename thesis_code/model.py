import lossfun 
import optimizer
import numpy as np

class Model():
    def __init__(self):
        self.layers = []


    def addLayer(self, layer):
        self.layers.append(layer)
    
    def setLoss(self, loss_function):
        loss = loss_function()
        self.lossfun = loss


    def forward(self, inputs):
        inputs = np.array(inputs)
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        for layer in self.layers:
            inputs = layer.forward(inputs, train=False)
        return inputs
    
    def backward(self, inputs, targets, learning_rate = 0.1, optimizer = optimizer.SGD):
        #Initialize optimizer
        opt = optimizer(learning_rate)

        #Transform inputs to needed form
        inputs = np.array(inputs)
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        #Transform targets to needed form
        targets = np.array(targets)
        if targets.shape == 1:
            targets = np.reshape(targets, (1, targets.shape))

        #Do a forward pass so that inputs of each layer is known
        for layer in self.layers:
            inputs = layer.forward(inputs, train=True)

        #Calculate the loss gradient
        loss_grad = self.lossfun.backward(inputs, targets)

        #The main operation is to find gradients of weights and biases for each of the neurons.
        #innerBackward returns the gradient of an input so we can backpropagate next(previous) layer.

        #Find the last layer gradient using loss gradient
        nextgrad = self.layers[len(self.layers) - 1].innerBackward(loss_grad)
        #Go through each layer and backpropagate the gradient
        #Order: last -> first layer
        for lay in range(len(self.layers)-2, -1, -1):
            #nextgrad - input gradient of next layer
            nextgrad = self.layers[lay].innerBackward(nextgrad)
        
        #Update parameters using optimizer
        for lay in self.layers:
            opt.forward(lay)


