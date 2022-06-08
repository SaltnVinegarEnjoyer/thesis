from audioop import avg
import layer
import model
import lossfun
import optimizer
import actfun
import os

import numpy as np



def mnist_train():
    #Initialize the model with crossenthropy loss
    mymodel = model.Model(lossfun.CrossEnthropy)

    #Create layers
    dense1 = layer.Dense(784, 15, actfun.ReLu)
    dense2 = layer.Dense(15, 20, actfun.ReLu)
    dense3 = layer.Dense(20, 10, actfun.Softmax)

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
    for image in range(len(mnist_data)):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.00001 - 0.99999
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99999) + 0.00001)
        #print(mnist_data[image])

    #Set the batch size
    batch_size = 50
    #Set the desired amount of epochs
    epochs_amount = 10

    # Train in loop
    for epoch in range(epochs_amount):

        accuracy_epoch = 0
        #This is a raw solution. I will change the last bit later on
        for imagen in range(0,(len(mnist_data)),batch_size):
            #Get the arrays of images and classes
            batch = mnist_data[imagen:imagen+batch_size]
            class_batch = classes[imagen:imagen+batch_size]

            #Check for last entry overflow
            if len(mnist_data) <= imagen+batch_size:
                batch = mnist_data[imagen:len(mnist_data)-1]
                class_batch = classes[imagen:len(classes)-1]

            #Get the network's predictions
            result = mymodel.forward(batch)
            #Get the most certain prediction
            predictions = np.argmax(result, axis=1)
            #Convert classes to numpy array
            target = np.array(class_batch)
            #Find the accuracy
            accuracy = np.mean(predictions==target) * 100
            #Add batch accuracy to epoch accuracy
            accuracy_epoch += accuracy


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
        #Print the information about an epoch pass
        print("Epoch: {}\tAccuracy in the last batch: {:.6f} %\tAverage accuracy in epoch: {:.2f} %\tLast batch loss: {}".format(epoch, accuracy, accuracy_epoch/(len(mnist_data)/batch_size), mymodel.calculate_loss(batch, target)))
        #Accuracy may be 100% while loss is not 0.
        #This is due to the network being "uncertain"

    test_model_mnist(mymodel)

    #Save the model
    mymodel.save_model('params.npy')

def test_model_mnist(model):
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
    for image in range(len(mnist_data)):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.00001 - 0.99999
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99999) + 0.00001)

    #Make a forward pass using all the numbers
    test_result = model.forward(mnist_data)
    #Convert the results to index form
    test_result = np.argmax(test_result, axis=1)
    #Find accuracy
    accuracy = np.mean(test_result == np.array(classes)) * 100

    #Print the testing results
    print("Accuracy in the test batch: {:.6f} %\tLoss: {:.6f}".format(accuracy, model.calculate_loss(mnist_data, classes)))

    #Accuracy may be 100% while loss is not 0.
    #This is due to the network being "uncertain"


def test_loading_model():

    #Initialize the model with crossenthropy loss
    mymodel = model.Model(lossfun.CrossEnthropy)

    #Create layers
    dense1 = layer.Dense(784, 15, actfun.ReLu)
    dense2 = layer.Dense(15, 20, actfun.ReLu)
    dense3 = layer.Dense(20, 10, actfun.Softmax)

    #Add layers to the model
    mymodel.addLayer(dense1)
    mymodel.addLayer(dense2)
    mymodel.addLayer(dense3)

    #Load parameters to the model
    mymodel.load_model('params.npy')

    test_model_mnist(mymodel)


def main():
    mnist_train()
    test_loading_model()


if __name__ == "__main__":
    main()