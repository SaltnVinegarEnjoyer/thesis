import numpy as np

class Model():
    def __init__(self, loss_function):
        #Initialize the array that will hold layer objects
        self.layers = []
        #Initialize the loss function
        self.lossfun = loss_function()

    def addLayer(self, layer):
        #Add layer to the array
        self.layers.append(layer)

    def forward(self, inputs):
        inputs = np.array(inputs)
        #If the inputs are just 1 sample outside the batch, add 1 more dimension
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        for layer in self.layers:
            #Do a forward pass on the neurons(x*w + b)
            layer.forward(inputs)
            #Memorize activation function's output for next step/returning the prediction
            inputs = layer.actfun.output
        #Return last layer's activation function's output as the network prediction
        return inputs
    
    #Get the loss from some sample
    def calculate_loss(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        return(self.lossfun.forward(self.forward(inputs), targets))
    
    def backward(self, inputs, targets):
        #Transform inputs to needed form(batched)
        inputs = np.array(inputs)
        if inputs.shape == 1:
            inputs = np.reshape(inputs, (1, inputs.shape))
        #Transform targets to needed form(batched)
        targets = np.array(targets)
        if targets.shape == 1:
            targets = np.reshape(targets, (1, targets.shape))
        
        #Do a forward pass so that inputs and outputs of each layer are memorized
        self.forward(inputs)

        #Calculate the loss gradient using output of the last layer's activation function and targets
        self.lossfun.backward(self.layers[len(self.layers) - 1].actfun.output, targets)

        #The main operation is to find gradients of weights and biases for each of the neurons.

        #Backpropagate through the last layer
        self.layers[len(self.layers) - 1].backward(self.lossfun.input_gradient)

        #Go through each layer and backpropagate the gradient
        #Order: last -> first layer
        for lay in range(len(self.layers)-2, -1, -1):
            #Backpropagate through layer using next layer's input gradient
            self.layers[lay].backward(self.layers[lay+1].input_gradient) 
        