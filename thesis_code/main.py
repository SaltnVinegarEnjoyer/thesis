import layer
import model
import lossfun
import optimizer
import actfun
import os

import numpy as np



def mnist():
    #Initialize the model with crossenthropy loss
    mymodel = model.Model(lossfun.CrossEnthropy)

    #Create layers
    dense1 = layer.Dense(784, 100, actfun.ReLu)
    dense2 = layer.Dense(100, 50, actfun.ReLu)
    dense3 = layer.Dense(50, 10, actfun.Softmax)

    #Add layers to the model
    mymodel.addLayer(dense1)
    mymodel.addLayer(dense2)
    mymodel.addLayer(dense3)

    # Create optimizer
    opt = optimizer.SGD(mymodel, learning_rate=1, decay=0.01, momentum=0.5)

    #Open the mnist train csv file
    mnist_file = open(os.sys.path[0] + "\\mnist_train.csv")
    #Read each line as row into the mnist_data array
    mnist_data = mnist_file.readlines()
    #Close the mnist file
    mnist_file.close()

    #Split each line(row(string) in mnist data) by comma
    for img in mnist_data:
        img = np.array(img.split(","))

    #Array of class values
    classes = []

    #Go through each image in mnist_data
    for image in range(len(mnist_data) - 1):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.01 - 0.99
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99) + 0.01)

    #Set the batch size
    test_batch = 50
    #Set the desired amount of epochs
    epochs_amount = 30

    # Train in loop
    for epoch in range(epochs_amount):

        #This is a raw solution. I will change the last bit later on
        for imagen in range(0,(len(mnist_data)-1) - test_batch,test_batch):
            #Get the arrays of images and classes
            batch = mnist_data[imagen:imagen+test_batch]
            class_batch = classes[imagen:imagen+test_batch]

            #Get the network's predictions
            result = mymodel.forward(batch)
            #Get the most certain prediction
            predictions = np.argmax(result, axis=1)
            #Convert classes to numpy array
            target = np.array(class_batch)
            #Find the accuracy
            accuracy = np.mean(predictions==target) * 100

            #Uncomment to see the amount of right values
            #strike = 0
            #for a,b in zip(predictions, target):
            #    if a == b:
            #        strike += 1
            #print("Amount of correct answers: ", strike, "/", bsize)

            #Backpropagate the results
            mymodel.backward(batch, target)

            #Optimize network's parameters
            opt.forward()
        #Print the last batch results in the epoch
        print("Epoch: ", epoch, "\tAccuracy in the last batch: ", int(accuracy), "%\tLoss: ", mymodel.calculate_loss(batch, target))
        #Accuracy may be 100% while loss is not 0.
        #This is due to the network being "uncertain"
    
    #Open the mnist test csv file
    mnist_file = open(os.sys.path[0] + "\\mnist_test.csv")
    #Read each line as row into the mnist_data array
    mnist_data = mnist_file.readlines()
    #Close the mnist file
    mnist_file.close()

    #Split each line(row(string) in mnist data) by comma
    for img in mnist_data:
        img = np.array(img.split(","))

    #Array of class values
    classes = []

    #Go through each image in mnist_data
    for image in range(len(mnist_data) - 1):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.01 - 0.99
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99) + 0.01)

    #Create the test batch
    test_batch = len(mnist_data) - 2

    #This is a raw solution. I will change the last bit later on
    for imagen in range(0,(len(mnist_data)-1) - test_batch,test_batch):
        #Get the arrays of images and classes
        batch = mnist_data[imagen:imagen+test_batch]
        class_batch = classes[imagen:imagen+test_batch]

        #Get the network's predictions
        result = mymodel.forward(batch)
        #Get the most certain prediction
        predictions = np.argmax(result, axis=1)
        #Convert classes to numpy array
        target = np.array(class_batch)
        #Find the accuracy
        accuracy = np.mean(predictions==target) * 100

    print("Accuracy in the test batch: ", int(accuracy), "%\tLoss: ", mymodel.calculate_loss(batch, target))
    #Accuracy may be 100% while loss is not 0.
    #This is due to the network being "uncertain"

    #Save the model
    mymodel.save_model('params.npy')


def test_loading_model():

    #Initialize the model with crossenthropy loss
    mymodel = model.Model(lossfun.CrossEnthropy)

    #Create layers
    dense1 = layer.Dense(784, 100, actfun.ReLu)
    dense2 = layer.Dense(100, 50, actfun.ReLu)
    dense3 = layer.Dense(50, 10, actfun.Softmax)

    #Add layers to the model
    mymodel.addLayer(dense1)
    mymodel.addLayer(dense2)
    mymodel.addLayer(dense3)

    #Load parameters to the model
    mymodel.load_model('params.npy')

    #Open the mnist test csv file
    mnist_file = open(os.sys.path[0] + "\\mnist_test.csv")
    #Read each line as row into the mnist_data array
    mnist_data = mnist_file.readlines()
    #Close the mnist file
    mnist_file.close()

    #Split each line(row(string) in mnist data) by comma
    for img in mnist_data:
        img = np.array(img.split(","))

    #Array of class values
    classes = []

    #Go through each image in mnist_data
    for image in range(len(mnist_data) - 1):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.01 - 0.99
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99) + 0.01)

    bsize = len(mnist_data) - 2

    #This is a raw solution. I will change the last bit later on
    for imagen in range(0,(len(mnist_data)-1) - bsize,bsize):
        #Get the arrays of images and classes
        batch = mnist_data[imagen:imagen+bsize]
        class_batch = classes[imagen:imagen+bsize]

        #Get the network's predictions
        result = mymodel.forward(batch)
        #Get the most certain prediction
        predictions = np.argmax(result, axis=1)
        #Convert classes to numpy array
        target = np.array(class_batch)
        #Find the accuracy
        accuracy = np.mean(predictions==target) * 100

    print("Accuracy in the test batch: ", int(accuracy), "%\tLoss: ", mymodel.calculate_loss(batch, target))


def main():
    mnist()
    test_loading_model()


if __name__ == "__main__":
    main()