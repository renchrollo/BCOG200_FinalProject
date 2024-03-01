# BCOG200_FinalProject
Independent Final Project for BCOG 200

**Project Description:**

My project aims to develop a simple Artificial Neural Network (ANN) for linear and non-linear regression tasks. This will be accomplished through developing a feedforward Neural Network structure and using Rectified Linear Unit (ReLU) activation functions for weighted sum calculations. 

While the nature of this network will be simplistic and not meant for processing and analyzing large amounts of data, I will take advantage of familiar libraries for smoother development.

Libraries: _**NumPy, Pandas, sci-kit-learn, PyTorch, matplotlib**_

*The listed libraries are tools that are relevant to small and large scale Neural Network development. Certain libraries denoted above might not be included in the final project, and the description will be adjusted as needed.


**Project Functions**

**1. Weighted Sum Function:**
   - A weighted sum function is the foremost function in any Neural Network. A weighted sum function provides a way of determining the strength of a connection between a previous neuron to the following neuron by applying a weight value. These values vary depending on the inputs, and determine the pattern of activation rates for one layer to the next, eventually impacting the neuron's output. During training processes, random weights are applied to neuron connections, and are slowly adjusted per epoch based on the desired output. As a result, we can shift the pattern of activations based on the input data, and improve the accuracy of the necessary pattern by shifting the weights.
   - Weighted Sum Function: a_1 = ReLU(W(a_0) + b), where a_1 represents the activation rate for a single neuron, W represents the weight value applied, a_0 represents the activation rate for the previous neuron, b represents the bias, and ReLU surrounds the output of all previous values.
   - This equation is a condensed representation of weighted sum calculations. In practice, the weights will be applied in a matrix and the previous activation rates and biases are applied in a vector. Calculations will be executed using the Dot Product.

  
**2. Activation Functions:**
   - Implementing an activation function in a Neural Network's hidden layer is critical for applying non-linear transformations to determine activation rates for complex and nonlinear data. This is essential in ensuring that the network acts less like a stacked linear regression model, and can handle more varied datasets.
   - ReLU activation function: f(x) = max(0, x), x represents the input, and the function outputs the larger value between the input and 0. If the input value is negative, the function will map that value to 0, and if positive, the function will map that value to the value itself. This is easier to implement than other popular functions like Sigmoid and tanH.

  
**3. Cost Functions:**
   - Developing a cost function is critical in any ANN training process. Cost functions provide a measure to the accuracy and consistency of a network by comparing the ANN output to the desired output. By adjusting the network's weights and biases to better predict the data, we can lower cost function and subsequently improve the accuracy of the model's output.
   - Cost function: f(x, y) = (x - y)^2, where x represents the actiavtion value of the output neuron, and y represents the desired activation value for the output neuron. This equation represents the cost function implementation for a single neuron, and must be repeated and combined into a vector depending on the number of neurons in the output layer. 

**_I'VE INCLUDED THE PRIMARY 3 FUNCTIONS NECESSARY IN NEURAL NETWORK DEVELOPMENT. THERE WILL BE MORE FUNCTIONS APPLIED TO THE FINAL PROJECT. I WILL INCLUDE THE FUNCTION DESCRIPTIONS IN THIS SECTION AS I CONTINUE BUILDING THE NETWORK._**
