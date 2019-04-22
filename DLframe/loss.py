"""
Loss functions are used to assess how good predictions of a model are.
There are different types of loss functions that have different characteristics
"""

import numpy as np

# Use our predefined tensor
from DLframe.tensor import Tensor

# Abstract base class - will be extended
class Loss:
    def loss(self, predictions, actual):
        # type: (Tensor, Tensor) -> float
        """
        Abstract base class for the loss function
        :param predictions: output from model
        :param actual: actual values/labels from data
        :rtype: float
        """
        raise NotImplementedError

    def grad(self, predictions, actual):
        # type: (Tensor, Tensor) -> Tensor
        """
        Abstract base class for the gradient of the loss function
        Needed to compute the gradients
        :param predictions:
        :param actual:
        :rtype: Tensor
        """
        raise NotImplementedError


class TSE(Loss):
    """
    Total squared error loss function.
    """
    def loss(self, predictions, actual):
        # type: (Tensor, Tensor) -> float
        """
        Total squared error is just the difference between the predictions
        and ground truth, squared
        :param predictions: output from model
        :param actual: actual values/labels from data
        :return: total squared error
        :rtype: float
        """
        return np.sum((predictions - actual) ** 2)

    def grad(self, predictions, actual):
        # type: (Tensor, Tensor) -> Tensor
        """
        Gradient is just the derivative of the loss function
        :param predictions: output from model
        :param actual: actual value/labels from data
        :rtype: Tensor
        """
        return 2 * (predictions - actual)


# TODO Implement more loss functions
