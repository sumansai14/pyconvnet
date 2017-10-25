from .baselayer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    r"""We first write the fully connectedlayer first and later abstract away the common parts to the baselayer."""

    def __init__(self, *args, **kwargs):
        r"""
        Fully Connected layer where dimensionality is automagically calculated.

        Unlike convnetjs, where the arrays are stored in 1d array,
        since we are using numpy, it becomes important to match the diemensions.

        For now, we leave the burden of dimensionality with users, just take the shape of the array
        """
        self.params = kwargs
        self.output_len = kwargs['num_neurons']
        self.input_len = kwargs['input_len']
        self.weights = np.random.randn(self.output_len, self.input_len)
        self.dw = np.zeros((self.output_len, self.input_len))
        self.biases = np.zeros((self.output_len, 1))
        self.db = None  # We don't actually need to store this since we are going to do the bias trick
        self.input_activations = None
        self.output_activations = None

    def forward(self, x, is_training):
        self.input_activations = x
        self.output_activations = np.dot(self.weights, x) + self.biases
        return self.output_activations

    # def backward(self):
        
