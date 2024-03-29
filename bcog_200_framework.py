import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# nnfs is a package that allows for random data sampling for testing ANNs.
# All input and output information is meant to draw patterns between data and highlight specific functions, not to analyze real data
# Any testing done by the course instructors can be accomplished through this package. Any other potential training data can be given to me.
class Layer_Comp:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # I initialized the weights randomly and multiplied all values by 0.10 to keep within the bounds of 0 and 1 to yield smaller overall activation values
        # The number of weights is determined by how many inputs from the previous layer matched to how many neurons in the current layer
        self.biases = np.zeros((1, n_neurons))
        # the shape is 1 by how many neurons we have, because biases are always a vector comprised of the number of neurons
        # when using np.zeros, we need 2 parentheses, with one the exterior most representing the parameters, and the inner on representing the shape
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # This is the weighted sum calculation for forward propagation
        # I used the dot product for matrix multiplication and followed the natural weighted sum equation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # I used the np.maximum function to find whether the inputs were greater than 0, returning either 0 if less than, and the input value if greater than

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Exponential Function
        # np.exp() finds the exponential the inputted values
        # np.max(inputs, axis=1, keepdims=True) computes the maximum value within each row (axis 1)
        # By subtracting the inputs, we prevent overflow errors common with exponentials to prevent crashing and unnecessarily large outputs
        prob_dist = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalization Function
        # np.sum(exp_values, axis=1, keepdims=True) takes the sum of exponentiated values along each row, as present in the softmax equation
        # By dividing the exp_value elements by the corresponding row sums, we ensure that the probability dist adds up to 1, or very close
        self.output = prob_dist

class Cost_Function:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        # Sample losses takes in the predictions (output) and the true, labeled values (y)
        # We take the forward method to calculate for every sample in the dataset
        data_loss = np.mean(sample_losses)
        # We take the mean of the sample losses to condense into a scalar value
        return data_loss

class Cost_Function_CCE(Cost_Function):
    # This function inherits all information and functions from the previous Cost Function class
    def forward(self, y_pred, y_true):
        samples = len(y)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # The natural logarithm of 0 is infinity, and in order to combat this, we must clip the within the bounds shown above
        # Any inputted values below 1e-7 are set to 1e-7, and any values above 1-1e-7 are set to 1-1e-7

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        # This if-else-elif block is meant to calculate confidence scores in both vector outputs (shape=1) and matrix outputs (shape=2)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
X, y = spiral_data(100, 3)

comp1 = Layer_Comp(2, 3)
activation1 = Activation_ReLU()

comp2 = Layer_Comp(3, 3)
activation2 = Activation_SoftMax()

comp1.forward(X)
activation1.forward(comp1.output)
comp2.forward(activation1.output)
activation2.forward(comp2.output)

print(activation2.output[:5])

cost_function = Cost_Function_CCE()
cost = cost_function.calc(activation2.output, y)

print("Cost:", cost)
