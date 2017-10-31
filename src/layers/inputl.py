from .baselayer import Layer
from .vector import Vector
import numpy as np


class InputLayer(Layer):

    def __init__(self, *args, **kwargs):
        r"""
        Input layer for the network.

        Need to implement dimensionality checks here so that it doesn't bite us
        in the ass later.
        :param x: array or numpy array with the weights
        """
        # check the type of array and change it to numpy array if it is an array
        self.params = kwargs
        self.input_len = kwargs.get('input_len')
        self.output_len = kwargs.get('input_len')
        self.dimensions = kwargs.get('dimensions')
        self.input_activations = Vector()
        self.output_activations = Vector()

    def forward(self, x, is_training):
        r"""
        We'll set some ground rules for dimensionality here.

        :param x: an nd array or dimensions(dimensions of image, num_inputs)
        """
        # Check the dimensionality of the input here
        # if type(x) == list:
        # x = np.array(x)
        assert ((x.shape[1], x.shape[2], x.shape[3]) == self.dimensions)
        self.input_activations.data = x
        self.output_activations.data = x
        return self.output_activations
