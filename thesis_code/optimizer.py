import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def forward(self, layer):
        #Simply substract derivative of weight and bias multiplied by the learning rate
        layer.weights = layer.weights - (self.lr * layer.weight_gradient)
        layer.biases = layer.biases - (self.lr * layer.bias_gradient)