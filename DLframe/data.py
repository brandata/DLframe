"""
Data is fed into the network in batches.
"""

import numpy as np
from DLframe.tensor import Tensor


class DataIterator:
    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, bs: int = 64, shuffle: bool = True) -> None:
        """
        Init the iterator
        :param bs: batch size
        :param shuffle: whether or not to shuffle the data
        """
        self.bs = bs
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor):
        starts = np.arange(0, len(inputs), self.bs)
        b = {}
        # Slight hack. If shuffle, shuffle all starting points
        if self.shuffle:
            np.random.shuffle(starts)

        # for each starting point, grab a batch of input/target
        # by indexing from start to start + bs
        for start in starts:
            end = start + self.bs
            b_in = inputs[start:end]
            b_targets = targets[start:end]
            yield b_in, b_targets
