from __future__ import absolute_import
from optimizers.minibatchgd import MiniBatchGradientDescent
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
    optimizer = MiniBatchGradientDescent(network)
    optimizer.train(train_x.T, train_y.T, 30, 30, learning_rate=3.0)
