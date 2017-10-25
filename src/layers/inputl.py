from .baselayer import Layer
import numpy as np


class InputLayer(Layer):

    def __init__(self, *args, **kwargs):
        r"""
        This is the input layer for the network.

        Need to implement dimensionality checks here so that it doesn't bite us
        in the ass later.
        :param x: array or numpy array with the weights
        """
        # check the type of array and change it to numpy array if it is an array
        self.params = kwargs
        self.input_len = kwargs['input_len']
        self.output_len = kwargs['input_len']
        self.input_activations = None
        self.output_activations = None

    def forward(self, x, is_training):
        # Check the dimensionality of the input here
        if type(x) == list:
            x = np.array(x)
        assert x.shape[0] == self.input_len
        self.input_activations = x
        self.output_activations = x
