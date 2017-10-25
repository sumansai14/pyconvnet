"""Definition of network class can be found in this file. WIll add a lot of todos later."""
from .layers import Layers
from exceptions import ValueError


class Network(object):
    r"""A neural network which takes in layer definitions and constructs the network."""

    def __init__(self, layers):
        r"""
        By default, all layers are linear, the non linearities are added as new layers.

        :params layers: a list of dictionaries containing the definition of layers
        """
        self.layers = []
        self.num_layers = len(layers)
        if layers[0]['type'] != Layers.INPUT:
            raise ValueError("the first layer must be input layer")
        for layer in layers:
            self.layers.append(layer['type'](**layer))
            if layer['activation']:
                self.layers.append(layer['activation'](**layer))

    def forward(self, x, is_training=False):
        r"""
        Take the input numpy array X, and an optinal boolean parameter is_training and does a forwardpass.

        :param X: numpy array of the inputs
        :param is_training: boolean field for the network to tell the layers whether or not to calculate the gradients
        """
        activation = self.layers[0].forward(x, is_training)
        for layer in self.layers[1:]:
            activation = layer.forward(activation, is_training)
        return activation
