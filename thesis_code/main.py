import layer
import actfun
import lossfun
import model
import numpy as np

def checkLossFunctionality():
    a = np.array([[2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8]])
    b = np.array([[3, -0.5, 2, 7],[3, -0.5, 2, 7]])
    loss = lossfun.MeanSquare()
    print("Mean square error loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", loss.forward(a, b))

    a = np.array([[0.1,1,0.9564], [0.1,0.1,1], [1,0.1,1], [0.56, 0.34, 0.9]])
    b = np.array([[0,0,1], [0,0,1], [0,0,1], [1,0,0]])
    loss = lossfun.CrossEnthropy()
    print("Cross enthropy(one-hot encoded) loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", loss.forward(a, b))

    a = np.array([[0.1,1,0.9564], [0.1,0.1,0], [1,0.1,1], [0.56, 0.34, 0.9]])
    b = np.array([2,2,2,0])
    print("Cross enthropy(index encoded) loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", loss.forward(a, b))


def checkModel():
    mdl = model.Model()
    mdl.setLoss(lossfun.CrossEnthropy)
    lay = layer.dense(3, 5, actfun.Relu)
    lay1 = layer.dense(5, 4, actfun.Softmax)
    mdl.addLayer(lay)
    mdl.addLayer(lay1)
    print("Forward result: ", mdl.forward([[1,2,3], [4,5,1]]))
    mdl.backward([[1,2,3],[4,5,1]], [0,2])
    print("Layer 1 weight gradient, shape: row - set of weight derivatives as per 1 neuron:\n", mdl.layers[0].weight_gradient)
    print("Layer 2 weight gradient, shape: row - set of weight derivatives as per 1 neuron:\n", mdl.layers[1].weight_gradient)
    #Train the network usning just 3 data samples
    for epoch in range(1000):
        mdl.backward([[1,2,3],[4,5,1], [3,3,3]], [0,2,0])
        print("Forward result: ", mdl.forward([[1,2,3], [4,5,6]]))
    #Notice the exponents value. Also, network now tries to predict 0th element 2 times harder than 2nd one

def main():
    checkModel()

if __name__ == "__main__":
    main()