import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Comp:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
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

class Loss:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(100, 3)

comp1 = Layer_Comp(2, 5)
activation1 = Activation_ReLU()

comp2 = Layer_Comp(5, 5)
activation2 = Activation_ReLU()

comp3 = Layer_Comp(5, 5)
activation3 = Activation_ReLU()

comp4 = Layer_Comp(5, 3)
activation4 = Activation_SoftMax()

comp1.forward(X)
activation1.forward(comp1.output)

comp2.forward(activation1.output)
activation2.forward(comp2.output)

comp3.forward(activation2.output)
activation3.forward(comp3.output)

comp4.forward(activation3.output)
activation4.forward(comp4.output)

print(activation4.output[:5])

loss_function = Loss_CCE()
loss = loss_function.calc(activation4.output, y)

print("Loss:", loss)