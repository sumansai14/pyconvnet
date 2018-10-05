from __future__ import absolute_import
from optimizers import MiniBatchGradientDescent
from utils import onehot
from network import Network
from datasets import MNISTDataSet
from layers import Layers, Activations
from estimator import Estimator
import numpy as np


def transform_data(x, y):
    x = x.reshape(x.shape[0], 784).T / 255
    y = onehot(y, 10).T
    # print(x.shape.T, y.shape)
    return x, y

if __name__ == '__main__':
    mnist = MNISTDataSet()
    layers = [
        {'type': Layers.INPUT, 'input_len': 784},
        {'type': Layers.FC, 'input_len': 784, 'num_neurons': 30, 'activation': Activations.SIGMOID},
        {'type': Layers.FC, 'input_len': 30, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    ]

    network = Network(layers)
    optimizer = MiniBatchGradientDescent(network, learning_rate=0.1, batch_size=30)
    estimator = Estimator(network=network, optimizer=optimizer, dataset=mnist, transformer=transform_data)
    estimator.train(epochs=30)
    estimator.test()
