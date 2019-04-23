
"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from DLframe.train import train
from DLframe.net import NeuralNet
from DLframe.layers import Linear, Tanh

if __name__ == '__main__':
    inputs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    targets = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    net = NeuralNet([
        Linear(input_size=2, output_size=2)
        # Tanh(),
        # Linear(input_size=2, output_size=2)
    ])

    train(net, inputs, targets)

    for x, y in zip(inputs, targets):
        predicted = net.forward(x)
    print(x, predicted, y)