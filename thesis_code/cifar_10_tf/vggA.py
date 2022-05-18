import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import datasets, Model
import numpy
import matplotlib.pyplot as plt
import time
from datetime import datetime


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 2
#Load the CIFAR10 dataset from the keras(will be changed later on)
#Now i want to repeatively create batched datasets due to lack of memory
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#Normalize the inputs to be on the scale 0.0-1.0
x_train, x_test = x_train / 255.0, x_test / 255.0
#Add another dimension for channels
#x_train = x_train[..., tf.newaxis].astype("float32")
#x_test = x_test[..., tf.newaxis].astype("float32")

#Shuffle the inputs and put them in batches of 32 images per batch
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
#Put test pictures in batches of 32. Shuffling will not take any effect, so we don't need to use it.
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

#Create the vgg-like model
class VGG_A(Model):
    def __init__(self):
      #This is really strange, but there's enough memory to add more convolutional layers, but not a single 4096 dence one
        #Call the parent class initialization
        super(VGG_A, self).__init__()
        #Architecture could be found here, in Table #1
        #https://arxiv.org/pdf/1409.1556.pdf
        
        #An array, where we will store VGG convolutional body
        self.vgg_body = []
        self.vgg_body.append(Conv2D(64, 3, activation='relu', padding="SAME"))
        self.vgg_body.append(MaxPooling2D((2,2)))
        self.vgg_body.append(Conv2D(128, 3, activation='relu', padding="SAME"))
        self.vgg_body.append(MaxPooling2D((2,2)))
        self.vgg_body.append(Conv2D(256, 3, activation='relu', padding="SAME"))
#        self.vgg_body.append(Conv2D(256, 3, activation='relu', padding="SAME"))
        self.vgg_body.append(MaxPooling2D((2,2)))
        self.vgg_body.append(Conv2D(512, 3, activation='relu', padding="SAME"))
#        self.vgg_body.append(Conv2D(512, 3, activation='relu', padding="SAME"))
        self.vgg_body.append(MaxPooling2D((2,2)))
        self.vgg_body.append(Conv2D(512, 3, activation='relu', padding="SAME"))
#        self.vgg_body.append(Conv2D(512, 3, activation='relu', padding="SAME"))
        self.vgg_body.append(MaxPooling2D((2,2)))
        
        #Array of latter VGG stages
        self.vgg_head = []
        self.vgg_head.append(Flatten())
        self.vgg_head.append(Dense(4096, activation='relu'))
        #The memory is just not enough to fit 1 last dense layer :c
        self.vgg_head.append(Dense(4096, activation='relu'))
        self.vgg_head.append(Dense(1000, activation='relu'))
        self.vgg_head.append(Dense(10, activation='softmax'))

    def call(self, x):
        for lay in self.vgg_body:
            x = lay(x)
        for lay in self.vgg_head:
            x = lay(x)
        return x

# Create an instance of the model
model = VGG_A()

#Create the loss object
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#Create the optimizer
optimizer = tf.keras.optimizers.SGD()


#The metrics that will show the loss during training process
train_loss = tf.keras.metrics.Mean(name='train_loss')
#The metrics that will show the batch accuracy during training process
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#The metrics that will show the loss during testing process
test_loss = tf.keras.metrics.Mean(name='test_loss')
#The metrics that will show the batch accuracy during testing process
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#Train the model 1 time using inputs and labels
@tf.function
def train_step(images, labels):
    #Gradient tape is used for automatic derrivation of the whole model
    with tf.GradientTape() as tape:
        #Do a forward pass. Remember the inputs on each step
        predictions = model(images, training=True)
        #Calculate the resulting loss
        loss = loss_object(labels, predictions)
    #Automatically calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    #Optimize the model using resulting gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #Get the loss after applying optimization
    train_loss(loss)
    #Get the accuracy after applying optimization
    train_accuracy(labels, predictions)

#Test the model
@tf.function
def test_step(images, labels):
  #Do a forward pass. We don't need to memorize new input values.
    predictions = model(images, training=False)
    #Calculate the resulting loss
    t_loss = loss_object(labels, predictions)
    #Get the loss
    test_loss(t_loss)
    #Get the accuracy
    test_accuracy(labels, predictions)


#Amount of epochs
EPOCHS = 5

CHECK_EVERY = 100

#Train the moden $EPOCHS times
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    batch_index = 0

    accuracy_history = []
    xdata = []
 
    plt.show()
    title = "Epoch " + str(epoch)
    plt.title(title)
    plt.xlabel("Batch number")
    plt.ylabel("Accuracy")
 
    #Define axes for plotting
    axes = plt.gca()
    axes.set_xlim(0, len(train_ds)/CHECK_EVERY)
    axes.set_ylim(0, 100)
    line, = axes.plot(xdata, accuracy_history, 'r-')

    #Let's divide the datasets into subdatasets, len/den each
    den = 200 #Subdatasets per dataset
    for subdataset_index in range((len(x_test)/den)-1):

        #Get subdatasets
        x_train_sub = x_train[subdataset_index*den:(subdataset_index+1)*den]
        y_train_sub = y_train[subdataset_index*den:(subdataset_index+1)*den]

        #Shuffle the inputs and put them in batches of 32 images per batch
        train_ds = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub)).shuffle(10000).batch(BATCH_SIZE)

        for images, labels in train_ds:
            train_step(images, labels)
            batch_index += 1
            #Update every n batches
            if batch_index % CHECK_EVERY == 0:
                #Get loss and accuracy values from the last batch processing
                loss = train_loss.result().numpy()
                accuracy = train_accuracy.result().numpy() * 100
                #Return the carriage, but don't go to the next line
                print("Epoch: ", epoch, "\tCurrent loss: {:.3f}".format(loss), "\tCurrent accuracy: {:.3f}%".format(accuracy), "\tImages processed: {:.0f}".format(batch_index * BATCH_SIZE), end="\r")
                #Append new values to memory
                accuracy_history.append(accuracy)
                xdata.append(int(batch_index/CHECK_EVERY))
                #Plot the updated data
                line.set_xdata(xdata)
                line.set_ydata(accuracy_history)
                plt.draw()
                #Wait for updating the plot
                plt.pause(0.1)
                time.sleep(0.1)
    #Close the plot for current epoch
    plt.close()

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_loss.result()}, '
      f'Accuracy: {train_accuracy.result() * 100}, '
      f'Test Loss: {test_loss.result()}, '
      f'Test Accuracy: {test_accuracy.result() * 100}'
    )
    # Save the weights
    dateTimeObj = datetime.now()
    checkpoint_name = "checkpoint_" + dateTimeObj.strftime("%d_%m_%Y__%H_%M_%S")
    model.save_weights('./checkpoints/' + checkpoint_name)








