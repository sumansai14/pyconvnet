import numpy as np


def onehot(arr, n):
    """
    convert an interable to an n-hot vector

    :param arr list: A list of numbers
    :param n int: N in the N-hot vector
    :return list: n-hot vectorised array
    """
    def vectorised(num):
        d = np.zeros((n, 1))
        d[num] = 1
        return d

    vector = np.array([vectorised(num) for num in arr])
    return np.reshape(vector, (vector.shape[0], vector.shape[1]))