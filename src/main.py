from __future__ import absolute_import
from network import Network
from datasets import MNISTDataSet
from layers import Layers, Activations

if __name__ == '__main__':
    train_x, train_y, validation_data, test_x, test_y = MNISTDataSet().data
    layers = [
        {'type': Layers.INPUT, 'input_len': train_x.shape[1]},
        {'type': Layers.FC, 'input_len': train_x.shape[1], 'num_neurons': 30, 'activation': Activations.SIGMOID},
        {'type': Layers.FC, 'input_len': 30, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    ]

    network = Network(layers)
    # mini_batches = [train_x[k:k + 10] for k in xrange(0, train_x.shape(0), 10)]
    a = network.forward(train_x[:20].T)
    b = network.backward(a, train_y[:20].T)
