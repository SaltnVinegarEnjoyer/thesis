import numpy as np


def meanSquare(result, target):
    return ((target - result) ** 2).mean()

def crossEnthropyOneHot(result, target):
    #There could be 2 types of CEL usage:
    #1 - We get a batch of one-hot encoded targets
    #2 - We get a vector of numbers, which represents the correct index of an array
    #For now, it is made only for one-hot encoded values

    #Target - one-hot matrix
    #Result - output of a layer

    #Get a 1-dimensional array with indexes of "hot"(1) values
    target_idx = np.argmax(target, axis=1)
    needed_vals =[]

    #Fill the vals array with results at "right" indexes
    for i in range(len(target_idx)):
        needed_vals.append(result[i, target_idx[i]])

    #Apply natural logarithm to the values. Smaller value -> bigger abs(output) 
    #Multiply the result by -1
    #Here may be an error because of taking log(0)
    loss = -1 * np.log(needed_vals)

    #Get an average value of array
    avg_loss = sum(loss) / len(loss)

    return avg_loss
