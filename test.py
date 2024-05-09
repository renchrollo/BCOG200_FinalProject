import numpy as np
import nnfs
from ANN_framework import Layer_Comp
from ANN_framework import Activation_ReLU
from ANN_framework import Activation_SoftMax
from ANN_framework import Loss_CCE

nnfs.init()
def test_Layer_Comp():
    layer = Layer_Comp(2, 5)
    assert layer.weights.shape == (2, 5)
    assert layer.biases.shape == (1, 5)

    layer = Layer_Comp(3, 3)
    assert layer.weights.shape == (3, 3)
    assert layer.biases.shape == (1, 3)

    layer = Layer_Comp(4, 2)
    assert layer.weights.shape == (4, 2)
    assert layer.biases.shape == (1, 2)

def test_Activation_ReLU():
    activation = Activation_ReLU()

    inputs = np.array([[1, -2], [-1, 2]])
    activation.forward(inputs)
    assert np.array_equal(activation.output, np.array([[1, 0], [0, 2]]))

    inputs = np.array([[0, 0], [0, 0]])
    activation.forward(inputs)
    assert np.array_equal(activation.output, np.array([[0, 0], [0, 0]]))

    inputs = np.array([[-1, -2], [-3, -4]])
    activation.forward(inputs)
    assert np.array_equal(activation.output, np.array([[0, 0], [0, 0]]))

def test_Activation_SoftMax():
    activation = Activation_SoftMax()

    inputs = np.array([[1, 2], [3, 4]])
    activation.forward(inputs)
    assert np.allclose(np.sum(activation.output, axis=1), 1)

    inputs = np.array([[0, 0], [0, 0]])
    activation.forward(inputs)
    assert np.allclose(np.sum(activation.output, axis=1), 1)

    inputs = np.array([[1, 1], [1, 1]])
    activation.forward(inputs)
    assert np.allclose(np.sum(activation.output, axis=1), 1)

def test_Loss_CCE():
    loss = Loss_CCE()

    y_pred = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    result = loss.forward(y_pred, y_true)
    assert np.allclose(result, np.array([0.35667494, 0.69314718]))

    y_pred = np.array([[1, 0, 0], [0, 1, 0]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    result = loss.forward(y_pred, y_true)
    assert np.allclose(result, np.full_like(result, 1e-7), atol=1e-7)

    y_pred = np.array([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3]])
    y_true = np.array([[0, 0, 1], [0, 1, 0]])
    result = loss.forward(y_pred, y_true)
    assert np.allclose(result, np.array([0.916290731874155, 0.916290731874155]))


if __name__ == "__main__":
    test_Layer_Comp()
    test_Activation_ReLU()
    test_Activation_SoftMax()
    test_Loss_CCE()
    print("All tests passed!")
