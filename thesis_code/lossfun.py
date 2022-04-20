import numpy as np


def meanSquare(result, target):
    return ((target - result) ** 2).mean()

def crossEnthropy(result, target):
    #Check if target values are one-hot encoded.
    #If it is, convert it to index vector form
    if len(target.shape) == 2:
        #Get a 1-dimensional array with indexes of "hot"(1) values
        target_idx = np.argmax(target, axis=1)
    elif len(target.shape) == 1:
        target_idx = target

    #Array of resulting values as per target
    needed_vals =[]

    #Fill the vals array with results at "right" indexes
    for i in range(len(target_idx)):
        needed_vals.append(result[i, target_idx[i]])

    #Apply natural logarithm to the values. Smaller value -> bigger abs(output) 
    #Multiply the result by -1
    #Here may be an error because of taking log(0)
    loss = -1 * np.log(needed_vals)

    #Get an average value of array
    avg_loss = np.mean(loss)

    return avg_loss
