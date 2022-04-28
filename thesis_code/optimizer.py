import numpy as np

class SGD:
    def __init__(self, model, learning_rate, decay = 0.01, momentum = 0.5):
        #Set the model that this obect is going to be working with
        self.model = model
        #Initialize starting learning rate
        self.learning_rate = learning_rate
        #Initialize actual learning rate
        self.lr = learning_rate
        #Set the decay
        self.decay = decay
        #Set the momentum
        self.momentum = momentum
        #Initialize "step" counter for updating the learning rate 
        self.step = 0

    def forward(self):
        #Decay the current learning rate
        #When decay is 0, it doesen't affect the learning rate at all
        self.lr = self.learning_rate * (1. / (1. + self.decay * self.step))
        #Go through each layer in the model
        for lay in self.model.layers:
            #Calculate and set new momentums for the layer
            #When momentum is 0, it doesen't affect the learning rate at all
            lay.weight_momentum = self.momentum * lay.weight_momentum - self.lr * lay.weight_gradient
            lay.bias_momentum = self.momentum * lay.bias_momentum - self.lr * lay.bias_gradient
            #Update layer's parameters based on the calculated momentums
            lay.weights += lay.weight_momentum
            lay.biases += lay.bias_momentum
        #Update step counter
        self.step += 1
    