from .baselayer import Layer
from .vector import Vector
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
        self.weights = Vector(data=np.random.randn(self.output_len, self.input_len))
        self.biases = Vector(data=np.zeros((self.output_len, 1)))
        self.input_activations = None
        self.output_activations = Vector()

    def forward(self, x, is_training):
        self.input_activations = x
        self.output_activations.data = np.dot(self.weights.data, x.data) + self.biases.data
        return self.output_activations

    def backward(self):

        self.weights.gradients = np.dot(self.output_activations.gradients, self.input_activations.data.T)
        self.biases.gradients = np.sum(self.output_activations.gradients, axis=1, keepdims=True)
        self.input_activations.gradients = np.dot(self.weights.T, self.output_activations.gradients)

    def get_params_and_grads(self):
        return [self.weights, self.biases]
