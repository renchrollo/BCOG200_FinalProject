# BCOG200_FinalProject
Independent Final Project for BCOG 200

**Project Description:**

My project aims to develop a simple Artificial Neural Network (ANN) for classification tasks and linear and non-linear regression tasks. This will be accomplished through developing a feedforward Neural Network structure and using Rectified Linear Unit (ReLU) activation functions for weighted sum calculations, and a Softmax activation function for probability distribution in outputs. 

While the nature of this network will be simplistic and not meant for processing and analyzing large amounts of data, I will take advantage of familiar libraries for smoother development.

Libraries: _**NumPy, nnfs, spiral_data**_

*The listed libraries are tools that are relevant to small and large scale Neural Network development. Certain libraries denoted above might not be included in the final project, and the description will be adjusted as needed.
- nnfs is a package that allows for random data sampling for testing ANNs.
- All input and output information is meant to draw patterns between data and highlight specific functions, not to analyze real data
- Any testing done by the course instructors can be accomplished through this package. Any other potential training data can be given to me.


**Project Functions**

**1. Weighted Sum Function:**
   - A weighted sum function is the foremost function in any Neural Network. A weighted sum function provides a way of determining the strength of a connection between a previous neuron to the following neuron by applying a weight value. These values vary depending on the inputs, and determine the pattern of activation rates for one layer to the next, eventually impacting the neuron's output. During training processes, random weights are applied to neuron connections, and are slowly adjusted per epoch based on the desired output. As a result, we can shift the pattern of activations based on the input data, and improve the accuracy of the necessary pattern by shifting the weights.
   - Weighted Sum Function: _a_1 = ReLU(W(a_0) + b)_, where a_1 represents the activation rate for a single neuron, W represents the weight value applied, a_0 represents the activation rate for the previous neuron, b represents the bias, and ReLU surrounds the output of all previous values.
   - This equation is a condensed representation of weighted sum calculations. In practice, the weights will be applied in a matrix and the previous activation rates and biases are applied in a vector. Calculations will be executed using the Dot Product.

  
**2. Activation Function: Hidden Layers:**
   - Implementing an activation function in a Neural Network's hidden layer is critical for applying non-linear transformations to determine activation rates for complex and nonlinear data. This is essential in ensuring that the network acts less like a stacked linear regression model, and can handle more varied datasets.
   - ReLU activation function: _f(x) = max(0, x)_, x represents the input, and the function outputs the larger value between the input and 0. If the input value is negative, the function will map that value to 0, and if positive, the function will map that value to the value itself. This is easier to implement than other popular functions like Sigmoid and tanH.

**3. Activation Function: Output Layer:**
   - In order to create a probability distribution and distinguish negative activation rates for the outputs, we must use a different activation function for the output layer. The function I will use in this network is the _Softmax Function_.
   - Softmax Function: _s(x_i) = (e^x_i)/(Σ_j=1->n e^x_j)_, this function is comprised of two parts: The Exponential Function and the Normalization Function
   - Exponential Function: _y = e^x_, e represents the base nat. logarithm, x represents the input to the neuron, y represents the output for the output layer neuron.
   - This function distinguishes negative outputs otherwise all assigned the value: 0 when passed through the ReLU activation function, where you can analyze negative outputs while making them distinguishable from positive outputs of the same value (can distinguish 9 from -9).
   - Normalization Function: _y = u/(Σi=1->n u_i)_, where u represents the output of the exponential function
   - This function yields the probability distribution with inputs from the exponential function.
  
**4. Cost Functions:**
   - Developing a cost function is critical in any ANN training process. Cost functions provide a measure to the accuracy and consistency of a network by comparing the ANN output to the desired output. By adjusting the network's weights and biases to better predict the data, we can lower cost function and subsequently improve the accuracy of the model's output.
   - In this ANN, I will calculate loss functions on two fronts: Loss on Given Predictions and Categorical Cross-Entropy loss for multi-class classification tasks.

