from __future__ import absolute_import
from optimizers.minibatchgd import MiniBatchGradientDescent
from network import Network
from datasets import MNISTDataSet
from layers import Layers, Activations

if __name__ == '__main__':
    train_x, train_y, validation_data, test_x, test_y = MNISTDataSet().data
    orig_shape = train_x.shape
    train_x = train_x.reshape(orig_shape[0], 1, 28, 28)
    layers = [
        {'type': Layers.INPUT, 'dimensions': (1, 28, 28)},
        {'type': Layers.CONV, 'stride': 1, 'fshape': (6, 1, 3, 3), 'padding': 1, 'activation': Activations.RELU},
        # {'type': Layers.CONV, 'input_len': 30, 'num_neurons': 10, 'activation': Activations.SIGMOID},
    ]
    train_x = train_x[:10, :, :, :]
    network = Network(layers)
    activation = network.forward(train_x, is_training=True)
    print(activation)
    # optimizer = MiniBatchGradientDescent(network)
    # optimizer.train(train_x.T, train_y.T, 30, 30, learning_rate=3.0)
