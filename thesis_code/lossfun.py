import numpy as np


#The gradients may be stored in the instances of loss functions themselves later on
#I still haven't descided the architecture yet

class MeanSquare:
    def forward(self, result, target):
        return ((target - result) ** 2).mean()

class CrossEnthropy:
    def forward(self, result, target):
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
  
        #Convert an array to numpy array
        needed_vals = np.array(needed_vals)
        #Replace all 0 values in the array to prevent division by 0
        needed_vals = np.where(needed_vals == 0, 1e-10, needed_vals)
        #Replace all 1(full match) values in the array to prevent overfloving (log(x) < 0 when x > 1)
        needed_vals = np.where(needed_vals == 1, 1 - 1e-10, needed_vals)
  
        #Apply natural logarithm to the values. Smaller value -> bigger abs(output) 
        #Multiply the result by -1
        loss = -1 * np.log(needed_vals)
  
        #Get an average value of array
        avg_loss = np.mean(loss)
  
        return avg_loss