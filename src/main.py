from __future__ import absolute_import
from optimizers import MiniBatchGradientDescent
from utils import onehot
from network import Network
from datasets import MNISTDataSet
from layers import Layers, Activations
import numpy as np

if __name__ == '__main__':
    mnist = MNISTDataSet()
    # train = mnist.train
    train_x, train_y = list(zip(*mnist.train))
    train_x = np.array(train_x)
    # 10 is the num of classes
    train_y = onehot(train_y, 10)
    train_x = train_x.reshape(train_x.shape[0], 784)
    print(train_x.shape, train_y.shape)
    layers = [
        {'type': Layers.INPUT, 'input_len': train_x.shape[1]},
        {'type': Layers.FC, 'input_len': train_x.shape[1], 'num_neurons': 30, 'activation': Activations.SIGMOID},
        {'type': Layers.FC, 'input_len': 30, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    ]

    network = Network(layers)
    optimizer = MiniBatchGradientDescent(network)
    optimizer.train(train_x.T, train_y.T, 30, 1000, learning_rate=1.0)
