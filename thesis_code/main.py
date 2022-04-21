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
    lay = layer.dense(3, 5, actfun.Relu)
    lay1 = layer.dense(5, 4, actfun.Softmax)
    mdl.addLayer(lay)
    mdl.addLayer(lay1)
    print(mdl.forward([[1,2,3], [4,5,6]]))

def main():
    checkLossFunctionality()
    checkModel()

    a = np.array([[1,1,1], [2,2,2], [3,3,3]])
    b = np.array([[1,3,0.5]])

    c = []
    gradset = np.sum(a, axis=0)
    print(gradset)


def fwfwefew():
    a = [1,2,3]
    c = []
    for i in a:
        gradset = np.sum(i, axis=1)
        print(gradset)
    #Reshape the gradient, since we have lost 1 dimension
        c.append(gradset.reshape(gradset.shape()[0], 1))
if __name__ == "__main__":
    main()