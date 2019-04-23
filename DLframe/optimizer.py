"""
Optimizers update the network based on the gradients
"""

from DLframe.net import NeuralNet


# Abstract Class
class Optimizer:
    def step(self, net):
        raise NotImplementedError


# Stochastic gradient decent
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        """
        The learning rate determines how much of the error to use when
        adjusting/updating the weights of the network
        :param lr: learning rate
        """
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        """
        Update the params by taking a scaled version of the gradients
        :param net: Network who's weights are being updated
        """
        for p, g in net.params_and_grads():
            p -= self.lr * g

# TODO Add more optimizers
