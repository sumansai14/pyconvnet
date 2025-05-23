"""Definition of network class can be found in this file. WIll add a lot of todos later."""
from __future__ import absolute_import
from pynet.layers import Layers
from pynet.loss import MSELoss


class Network(object):
    r"""A neural network which takes in layer definitions and constructs the network."""

    def __init__(self, layers, loss=MSELoss):
        r"""
        By default, all layers are linear, the non linearities in form of activations are added as new layers.

        :params layers: a list of dictionaries containing the definition of layers
        """
        self.layers = []
        self.loss_function = loss()
        if layers[0]['type'] is not Layers.INPUT:
            raise ValueError("the first layer must be input layer")
        for idx, layer in enumerate(layers):
            self.layers.append(layer['type'].value(**layer))
            if layer.get('activation'):
                self.layers.append(layer['activation'].value(**layer))
        self.num_layers = len(self.layers)

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

    def backward(self, x, y):
        self.loss_function.backward(y)
        # Layerwise loss
        for l in range(1, self.num_layers):
            self.layers[-l].backward()
        return None

    def loss(self, x, y):
        accuracy, loss = self.loss_function.forward(x, y)
        return accuracy, loss

    def get_params_and_grads(self):
        params = []
        for idx, layer in enumerate(self.layers):
            # print(idx)
            params += layer.get_params_and_grads()
        return params
