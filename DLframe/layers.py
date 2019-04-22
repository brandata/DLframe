"""
Layers apply functions/gradients to tensors and are often stacked together
where the output of one layer forms the input to another
"""

import numpy as np
from DLframe.tensor import Tensor


# Abstract base class for layers
class Layer:
    def __init__(self):
        self.params = {}  # dict of weights and bias terms
        self.grads = {}  # dict of gradients

    def forward(self, inputs: Tensor) -> Tensor:
        """
        A forward pass produces outputs from the given inputs
        :param inputs: The input tensor
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        A backward pass passes the gradients from one layer to the previous
        layer
        :param grad: The tensor of gradients
        """
        raise NotImplementedError


class Linear(Layer):
    """
    A linear layer computs outputs as a linear combination of its inputs
    """

    def __init__(self, input_size, output_size):
        """
        Override init function
        :param input_size: (batch_size, input_dim)
        :param output_size: (batch_size, output_dim)
        """
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)  # weights
        self.params["b"] = np.random.rand(output_size)  # bias

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The output of a layer is calculated by computing the matrix multiplication
        between the inputs and weights, summed with a bias term
        :param inputs: The input tensor
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]  # matrix multiply

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass passes the computed gradients from one layer
        to the layer previous to it
        :param grad: Gradient tensor
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs @ grad
        return grad @ self.params["w"].T


class Activation(Layer):
    """
    Activation layers apply functions to each cell in the corresponding
    tensor
    """

    def __init__(self, f: function, f_prime: function) -> None:
        """
        Override init. This takes a function and its derivative and applies
        it to all cells in the tensor
        :param f: activation function
        :param f_prime: derivative of activation function
        """
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply the function to each input elementwise
        :param inputs: input tensor
        """
        self.inputs = inputs  # save the inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Apply the derivative of the activation function to the inputs
        elementwise
        :param grad: gradient tensor
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    return 1 - tanh(x) ** 2  # derivative of tanh function


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime())
