# tensorflow_util.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import tensorflow as tf
import util

_SESSION = None


def get_session():
    global _SESSION
    if _SESSION is None:
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        # _SESSION = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        _SESSION = tf.InteractiveSession()
    return _SESSION


def cleanup_session():
    global _SESSION
    del(_SESSION)
    _SESSION = None


class MinibatchIndefinitelyGenerator(object):
    def __init__(self, data, batch_size, shuffle):
        """Generator to iterate through all the data indefinitely in batches.

        Args:
            data: list of numpy.arrays, where each of them must be of the same
                length on its leading dimension.
            batch_size: number of data points to return in a minibatch. It must
                be at least 1.
            shuffle: if True, the data is iterated in a random order, and this
                order differs for each pass of the data.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be a positive integer, %r given" % batch_size)
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        # total number of data points in data
        self._N = None
        for datum in self._data:
            if self._N is None:
                self._N = len(datum)
            else:
                if self._N != len(datum):
                    raise ValueError("data have different leading dimensions: %d and %d" % (self._N, len(datum)))
        if self._shuffle:
            self._fixed_random = util.FixedRandom()
        self._indices = np.array([], dtype=int)

    def next(self):
        """Returns the next minibatch for each of the numpy.arrays in the data list.

        Returns:
            a tuple of batches of the data, where each batch in the tuple comes
                from each of the numpy.arrays data provided in the constructor.

        Examples:
            >>> gen = MinibatchIndefinitelyGenerator([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])], 1, False)
            >>> print(gen.next())
            (array([1]), array([4]), array([7]))
            >>> print(gen.next())
            (array([2]), array([5]), array([8]))
            >>> gen = MinibatchIndefinitelyGenerator([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])], 2, False)
            >>> print(gen.next())
            (array([1, 2]), array([4, 5]), array([7, 8]))
            >>> print(gen.next())
            (array([3, 1]), array([6, 4]), array([9, 7]))
        """
        if len(self._indices) < self._batch_size:
            new_indices = np.arange(self._N)
            if self._shuffle:
                self._fixed_random.random.shuffle(new_indices)
            self._indices = np.append(self._indices, new_indices)
        excerpt, self._indices = np.split(self._indices, [self._batch_size])
        data_batch = tuple(datum[excerpt] for datum in self._data)
        return data_batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
