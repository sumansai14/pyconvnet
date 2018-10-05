from .baselayer import Layer
from .vector import Vector
import numpy as np


class FlattenLayer(Layer):
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
        self.weights = Vector(data=np.zeros((self.output_len, self.input_len)))
        self.biases = Vector(data=np.zeros((self.output_len, 1)))
        self.input_activations = Vector()
        self.output_activations = Vector()

    def forward(self, x, is_training):
        print(x.data.shape)
        if len(x.data.shape) == 4:
            # This is a hack - this is coming from a conv. Flatten it.
            self.input_activations.data = x.data.reshape(x.data.shape[0], x.data.shape[1])
        self.output_activations.data = self.input_activations.data
        return self.output_activations

    def backward(self):

        self.weights.gradients = np.zeros(self.weights.data.shape)
        self.biases.gradients = np.zeros(self.biases.data.shape)
        self.input_activations.gradients = self.output_activations.gradients

    def get_params_and_grads(self):
        return [self.weights, self.biases]