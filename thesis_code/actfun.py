import numpy as np
import math

def sigmoid(arr):
    #This is just an example. I am going to make a more optimized implementation later on.
    arr = np.array(arr)
    #Remember the original shape
    dims = np.shape(arr)
    #Reshape an array to 1D
    arr = arr.reshape(-1)

    #Loop through each of the element and apply expit to it
    for i in range(len(arr)):
        arr[i] = (1/(1+math.exp(-arr[i])))

    #Reshape back to original shape
    arr = arr.reshape(dims)
    return arr

#max(0, x)
def relu(x):
    x = np.maximum(0, x)
    return x

def softmax(x):
    #Get an exponintiated array
    exponents = np.exp(x)
    #Get the dimensions 
    dims = np.shape(x)
    #Get a vector of sums(as per output set)
    expsum = np.sum(exponents, axis=1)
    #Reshape the output, so that each value is a vector
    expsum = expsum.reshape(dims[0], 1)

    #Divide exponentiated values by the sum of each exponintiated output set(dim 1)
    norm = exponents / expsum
    return norm