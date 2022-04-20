import layer
import actfun
import lossfun
import numpy as np

def checkFunctionality():
    a = np.array([[2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8]])
    b = np.array([[3, -0.5, 2, 7],[3, -0.5, 2, 7]])
    print("Mean square error loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", lossfun.meanSquare(a, b))

    a = np.array([[0.1,1,0.9564], [0.1,0.1,1], [1,0.1,1], [0.56, 0.34, 0.9]])
    b = np.array([[0,0,1], [0,0,1], [0,0,1], [1,0,0]])
    print("Cross enthropy(one-hot encoded) loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", lossfun.crossEnthropy(a, b))

    a = np.array([[0.1,1,0.9564], [0.1,0.1,1], [1,0.1,1], [0.56, 0.34, 0.9]])
    b = np.array([2,2,2,0])
    print("Cross enthropy(index encoded) loss:")
    print("Out: \n", a)
    print("Target: \n", b)
    print("Result: \n", lossfun.crossEnthropy(a, b))

def main():
    checkFunctionality()
    print(-np.log(1 - 1e-8))

if __name__ == "__main__":
    main()