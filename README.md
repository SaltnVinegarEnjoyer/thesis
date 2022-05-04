This is a repository that contains the code for my final thesis  
Link to the MNIST dataset in csv format: https://pjreddie.com/projects/mnist-in-csv/

# Structure
1. Model class, which is responcible for connecting all the parts of the code together.
2. Layer class, which is responsible for processing the inputs using internal logic. Also calls the activation function.
3. Activation function, which is responsible for taking inputs from the layer and apply special formula to produse the output.
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