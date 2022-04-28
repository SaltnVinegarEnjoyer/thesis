import numpy as np

class SGD:
    def __init__(self, model, learning_rate, decay = 0):
        #Set the model that this obect is going to be working with
        self.model = model
        #Initialize starting learning rate
        self.learning_rate = learning_rate
        #Initialize actual learning rate
        self.lr = learning_rate
        #Set the decay
        self.decay = decay
        #Initialize "step" counter for updating the learning rate 
        self.step = 0

    def forward(self):
        #Decay the current learning rate
        #When decay is 0, it doesen't affect the learning rate at all
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.step))
        #Go through each layer in the model
        for lay in self.model.layers:
            #Simply substract gradients of weight and bias multiplied by the learning rate
            lay.weights += (-self.current_learning_rate * lay.weight_gradient)
            lay.biases += (-self.current_learning_rate * lay.bias_gradient)
        #Update step counter
        self.step += 1
    