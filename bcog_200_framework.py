import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Comp:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # the shape is 1 by how many neurons we have, because biases are always a vector comprised of the number of neurons
        # when using np.zeros, we need 2 parentheses, with one the exterior most representing the parameters, and the inner on representing the shape
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob_dist = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob_dist

class Loss_Function:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Function_CCE(Loss_Function):
    def forward(self, y_pred, y_true):
        samples = len(y)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

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

loss_function = Loss_Function_CCE()
loss = loss_function.calc(activation2.output, y)

print("Loss:", loss)
