"""
A neural network is comprised of several layers concatenated together.
There can be an arbitrary number and can be of arbitrary type. The output
of one layer serves as the input to another.
"""

from DLframe.tensor import Tensor
from DLframe.layers import Layer


# Abstract Class
class NeuralNet:
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The forward pass of the network. Outputs of one layers are passed
        as inputs to the next
        :param inputs: Tensor of inputs (usually prev layer outputs)
        :return: Tensor
        """
        # Iterate through layers and pass the inputs along
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass passes the gradients backwards. Ie the error
        in this layer is a function of the errors that it caused later in
        the network
        :param grad:
        :return:
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        """
        Yields the params and gradients for all the layers
        """
        for layer in self.layers:
            for n, p in layer.params.items():
                grad = layer.grads[n]
                yield p, grad
