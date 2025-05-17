from __future__ import absolute_import
from pyconvnet.optimizers import MiniBatchGradientDescent
from pyconvnet.utils.transforms import onehot
from pyconvnet.network import Network
from pyconvnet.datasets import MNISTDataSet
from pyconvnet.layers import Layers, Activations
from pyconvnet.estimator import Estimator
import numpy as np


def transform_data(x, y):
    x = np.reshape(x, (x.shape[0], 784))
    x = x.T / 255
    y = onehot(y, 10).T
    # print(x.shape, y.shape)
    return x, y

def transform_data_conv(x, y):
    x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
    y = onehot(y, 10)
    return x, y

if __name__ == '__main__':
    mnist = MNISTDataSet()
    # layers = [
    #     {'type': Layers.INPUT, 'input_len': 784},
    #     {'type': Layers.FC, 'input_len': 784, 'num_neurons': 30, 'activation': Activations.SIGMOID},
    #     {'type': Layers.FC, 'input_len': 30, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    # ]

    layers = [
        {'type': Layers.INPUT, 'dimensions': (1, 28, 28)},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (6, 1, 3, 3), 'padding': 1, 'activation': Activations.RELU},
        {'type': Layers.MAXPOOL, 'stride': 3, 'length': 3},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (12, 6, 3, 3), 'padding': 1, 'activation': Activations.RELU},
        {'type': Layers.MAXPOOL, 'stride': 3, 'length': 3},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (24, 12, 3, 3), 'padding': 1, 'activation': Activations.RELU},
        {'type': Layers.MAXPOOL, 'stride': 3, 'length': 3},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (24, 24, 1, 1), 'padding': 0, 'activation': Activations.RELU},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (10, 24, 1, 1), 'padding': 0, 'activation': Activations.RELU},
        {'type': Layers.FLATTEN, 'input_len': 24, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    ]

    network = Network(layers)
    optimizer = MiniBatchGradientDescent(network, learning_rate=0.01, batch_size=16)
    estimator = Estimator(network=network, optimizer=optimizer, dataset=mnist, transformer=transform_data_conv)
    estimator.train(epochs=30)
    estimator.test()
