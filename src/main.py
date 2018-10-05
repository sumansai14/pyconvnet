from __future__ import absolute_import
from optimizers import MiniBatchGradientDescent
from utils import onehot
from network import Network
from datasets import MNISTDataSet
from layers import Layers, Activations
from estimator import Estimator
import numpy as np
import ipdb as pdb


def transform_data(x, y):
    # print(x.shape)
    # print(x[0])
    # x = x.reshape(784, x.shape[0]).T.reshape(784, x.shape[0]) / 255
    # print(x[0].shape)
    # x = x.T
    # print(x.shape)
    # print(x[0].shape)
    # print(x[0])
    # pdb.set_trace()
    # y =
    # print(x.shape)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
    y = onehot(y, 10)
    # print(y.shape)
    # y = np.reshape(y, (y.shape[0], y.shape[1], 1, 1))
    # print(x.shape, y.shape)
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
    optimizer = MiniBatchGradientDescent(network, learning_rate=0.1, batch_size=30)
    estimator = Estimator(network=network, optimizer=optimizer, dataset=mnist, transformer=transform_data)
    estimator.train(epochs=30)
    estimator.test()
