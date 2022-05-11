This is a repository that contains the code for my final thesis  .
Link to the MNIST dataset in csv format: https://pjreddie.com/projects/mnist-in-csv/

# Structure
1. Model class, which is responcible for connecting all the parts of the code together.
2. Layer class, which is responsible for processing the inputs using internal logic. Also calls the activation function.
3. Activation function, which is responsible for taking inputs from the layer and apply special formula to produce the output.
4. Loss function, which is used at the end of the neural network structure. It is used to calculate, how much of a deviation the model has made.
5. Optimizer. It is used for updating the parameters of the model.
# Order of the operation
## Description of the forward pass(using dense layers)
1. Model gets an inputs. I passes it through all the layers, which are stored sequentially in the `self.layers` property. It takes `self.layers.actfun.output` as the "layer's output".
2. Each layer gets a new set of inputs, computes the outputs(takes a dot product of inputs and weight. Then adds the biases). After that the output is passed to the layer's activation function.
3. Activation function applies its logic and saves the output in the `self.output`.
4. The model returns the final layer's activation function's output.
## Description of the backward pass(using dense layers)
1. Model gets an inputs and outputs as batches. It does the forward pass first to memorize inputs and outputs of every detail it consists of.
2. The model computes the loss's input gradient using last layer's activation function's output and targets as inputs to the loss function.
3. The model runs bakward operation on the last layer using the loss's input gradient.
4. The layer finds the activation function's input gradient first by passing rescieved gradient to the activation function's backwards method. Then it uses the resulting matrix to compute derivatives of every single parameter it has as compared to the activation function's gradient(finds how does changing a parameter affects the resulting loss).
5. After that, it computes the self input gradient, which will be used in the preceding layer.
6. The model reversely loops through all the layers it has, except the last one. It uses next layer's input gradient as the gradient that is passed to the preceiding layer.
## Desctiption of the SGD optimization step
1. The learning rate is modified using the decay.
2. Optimizer loops through each of the model's layers, calculates the weights and biases momentums, and adds them to the weights and biases themselves(in order to decrease the resulting loss).

# Annotation tools
## automatic_annotation.py
This script is used to produce labels automatically. It uses YOLOv3-608 as the base model. The model is loaded and run using opencv's dnn module. Since the resolution of original is too huge, i have descided to split each frame into subframes and then feed these parts of the image to the model. After each subframe is processed, i am compiling everything back to fit original frame. I may save annotations as per subframe in the future, since it is most likely that the self-made model will be trained using only parts of the image.
## static_label_propogation.py
This script is used to propagate bounding boxes of static objects through all the frames. It simply reads all the labels from the premade file `static_labels.txt` and adds them to the end of each frame's label file. It will require improvements if i will decide to work with subframes later on. The `classes.txt` file is ignored during the process.
## Ready-made tools used
I have descided to use [LabelImg](https://github.com/tzutalin/labelImg) for manual annotation. It is very lightweight.

# Annotation workflow
1. Get the video samples(yt-dl, etc.)
2. Automatically annotate most of the labels using automatic_annotation.py
3. Make a copy of first frame's annotation file. Open the first frame in LabelImg. Delete all the automatically annotated labels. Annotate every static object(you may visually check if an object have moved using first and last frames as reference). Rename filled file to static_labels.txt. Rename back the first frame's autommaticaly annotated file.
4. Run static_label_propogation.py
5. Add/remove/adjust annotations using LabelImg.