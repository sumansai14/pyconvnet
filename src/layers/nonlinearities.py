from .baselayer import Layer
from .vector import Vector
import numpy as np


class SigmoidLayer(Layer):

    def __init__(self, *args, **kwargs):
        # self.output_len = kwargs['num_neurons']
        self.input_activations = None
        self.output_activations = Vector()

    def forward(self, x, is_training):
        self.input_activations = x
        self.output_activations.data = 1 / (1 + np.exp(-(x.data)))
        return self.output_activations

    def backward(self):
        self.input_activations.gradients = self.output_activations.gradients * (self.output_activations.data * (1 - self.output_activations.data)) # noqa
