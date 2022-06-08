from audioop import avg
import layer
import model
import lossfun
import optimizer
import actfun
import os

import numpy as np



def mnist_train(model_to_use=0, learning_rate=0.5):
    #If model is not provided
    if model_to_use == 0:
        #Initialize the model with crossenthropy loss
        model_to_use = model.Model(lossfun.CategoricalCrossEntropy)

        #Create layers
        dense1 = layer.Dense(784, 15, actfun.ReLu)
        dense2 = layer.Dense(15, 20, actfun.ReLu)
        dense3 = layer.Dense(20, 10, actfun.Softmax)

        #Add layers to the model
        model_to_use.addLayer(dense1)
        model_to_use.addLayer(dense2)
        model_to_use.addLayer(dense3)

    # Create optimizer
    opt = optimizer.SGD(model_to_use, learning_rate=learning_rate, decay=0.01, momentum=0.5)

    #Open the mnist train csv file
    mnist_file = open(os.sys.path[0] + "\\mnist_train.csv")
    #Read each line as row into the mnist_data array
    mnist_data = mnist_file.readlines()
    #Close the mnist file
    mnist_file.close()

    #Split each line(row(string) in mnist data) by comma
    for img in mnist_data:
        img = np.array(img.split(","))

    #Array of class(target) values
    classes = []

    #Go through each image in mnist_data
    for image in range(len(mnist_data)):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #Append new image(indexes 1:785)
        #We want to have neural network's inputs in the range of 0.00001 - 0.99999
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99999) + 0.00001)

    #Set the batch size
    batch_size = 50
    #Set the desired amount of epochs
    epochs_amount = 10

    #Find amount of batches
    batch_amount = (len(mnist_data)/batch_size)

    # Train in loop
    for epoch in range(epochs_amount):

        #Variable for calculating accuracy as per whole epoch
        accuracy_epoch = 0
        #Variable for calculating loss as per whole epoch
        loss_epoch = 0

        #Go through all images using step of batch size
        for image_number in range(0,(len(mnist_data)),batch_size):
            #Get the arrays of images and classes
            batch = mnist_data[image_number:image_number+batch_size]
            class_batch = classes[image_number:image_number+batch_size]

            #Check for last entry overflow
            if len(mnist_data) <= image_number+batch_size:
                batch = mnist_data[image_number:len(mnist_data)-1]
                class_batch = classes[image_number:len(classes)-1]

            #Get the network's predictions
            result = model_to_use.forward(batch)
            #Get the most certain prediction
            predictions = np.argmax(result, axis=1)
            #Convert classes to numpy array
            target = np.array(class_batch)
            #Find the accuracy
            accuracy = np.mean(predictions==target) * 100
            #Add batch accuracy to epoch accuracy
            accuracy_epoch += accuracy


            #Uncomment to see the amount of correct guesses
            #strike = 0
            #for a,b in zip(predictions, target):
            #    if a == b:
            #        strike += 1
            #print("Amount of correct answers in batch: ", strike, "/", bsize)

            #Find the gradient
            model_to_use.backward(batch, target)

            #Add loss to the whole loss
            loss_epoch += model_to_use.calculate_loss(batch, target)


            #Optimize network's parameters
            opt.forward()
        #Print the information about an epoch pass
        print("Epoch: {}\tAccuracy in the last batch: {:.3f} %\tAverage accuracy in epoch: {:.3f} %\tLast batch loss: {:.5f}\tAverage loss in epoch: {:.5f}".format\
            (epoch, accuracy, accuracy_epoch/batch_amount, model_to_use.calculate_loss(batch, target),loss_epoch/batch_amount))
        #Accuracy may be 100% while loss is not 0.
        #This is due to the network being "uncertain"
    
    #Return the trained model
    return model_to_use

def test_model(model):
    #Open the mnist test csv file
    mnist_file = open(os.sys.path[0] + "\\mnist_test.csv")
    #Read each line as row into the mnist_data array
    mnist_data = mnist_file.readlines()
    #Close the mnist file
    mnist_file.close()

    #Split each line(row(string) in mnist data) by comma
    for img in mnist_data:
        img = np.array(img.split(","))

    #Array of class(target) values
    classes = []

    #Go through each image in mnist_data
    for image in range(len(mnist_data)):
        #Append new class value(see the mnist dataset structure)
        classes.append(int(mnist_data[image][0]))
        #We want to have neural network's inputs in the range of 0.00001 - 0.99999
        mnist_data[image] = ((np.asfarray(mnist_data[image].split(",")[1:]) / 255.0 * 0.99999) + 0.00001)

    #Make a forward pass using all the images
    test_result = model.forward(mnist_data)
    #Convert the results to index form
    test_result = np.argmax(test_result, axis=1)
    #Find accuracy
    accuracy = np.mean(test_result == np.array(classes)) * 100

    #Print the testing results
    print("Accuracy in the test batch: {:.6f} %\tLoss: {:.6f}".format(accuracy, model.calculate_loss(mnist_data, classes)))

    #Accuracy may be 100% while loss is not 0.
    #This is due to the network being "uncertain"


def load_model(filename='params.npy'):
    #Loading the model may be encapsulated to another layer of complexity, but i think it is enough for the demonstration purpouses

    #Initialize the model
    loaded_model = model.Model(lossfun.CategoricalCrossEntropy)

    #Create layers
    dense1 = layer.Dense(784, 15, actfun.ReLu)
    dense2 = layer.Dense(15, 20, actfun.ReLu)
    dense3 = layer.Dense(20, 10, actfun.Softmax)

    #Add layers to the model
    loaded_model.addLayer(dense1)
    loaded_model.addLayer(dense2)
    loaded_model.addLayer(dense3)

    #Load parameters to the model
    loaded_model.load_model(filename)
    
    #Return the loaded model
    return loaded_model

    #Test if model performs as intended
    test_model(loaded_model)


def main():
    #Create and train a model
    model = mnist_train()
    #Do a test on a model using mnist_test dataset
    test_model(model)
    #Train a model 2nd time
    model = mnist_train(model, learning_rate=0.01)
    #Do a test on a model using mnist_test dataset
    print("Testing the final model:")
    test_model(model)
    #Save the model to a file
    model.save_model('mnist_model_params.npy')
    #Load the model from a file
    loaded_model = load_model('mnist_model_params.npy')
    #Do a test on a loaded model using mnist_test dataset
    print("Testing the loaded model:")
    test_model(loaded_model)


if __name__ == "__main__":
    main()