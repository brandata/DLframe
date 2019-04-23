"""
This comprises the main training loop
"""

from DLframe.tensor import Tensor
from DLframe.net import NeuralNet
from DLframe.loss import Loss, TSE
from DLframe.optimizer import Optimizer, SGD
from DLframe.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 2000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          optimizer: Optimizer = SGD()) -> None:
    """
    Training loop. Loop through each epoch do passes (forward/backward),
    calculate error/grads, update weights, continue
    :param net: Full neural network
    :param inputs: Input tensor
    :param targets: training labels
    :param num_epochs: number of training loops to do
    :param iterator: which type of data iterator to use
    :param loss: which type of loss function to use
    :param optimizer: which type of optimizer to use
    """
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inp, targ in iterator(inputs, targets):
            predictions = net.forward(inp)
            epoch_loss += loss.loss(predictions, targ)
            grad = loss.grad(predictions, targ)
            net.backward(grad)
            optimizer.step(net)
        if epoch % 100 == 0:
            print(epoch, epoch_loss)
