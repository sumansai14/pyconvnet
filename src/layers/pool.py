from .baselayer import Layer
from .vector import Vector
import numpy as np


class MaxPool(Layer):

    def __init__(self, *args, **kwargs):
        self.length = kwargs['length']
        self.stride = kwargs['stride']
        self.input_activations = None
        self.output_activations = Vector()

    def forward(self, x, is_training):
        r"""
        A Very naive implementation of maxpool.

        Maxpool does subsampling. What that means is, for every 2*2 matrix (if length is 2 i.e.) there will be a
        a single element which is either mean, l2 mean, or max (which is currently the case) of these 4 elements
        effectively taking the most important activations in a sample.
        :param x: input activations for the layer
        :param is_training: true/false to denote whether or not we are doing training.
        returns subsampled input activations.
        """
        self.input_activations = x
        n, c, h1, w1 = x.data.shape
        s = self.stride
        w2 = 1 + ((w1 - self.length) / s)
        h2 = 1 + ((h1 - self.length) / s)
        data = np.zeros((n, c, w2, h2))
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                for h in range(0, x.shape[2], s):
                    for w in range(0, x.shape[3], s):
                        data[n, c, int(h / s), int(w / s)] = np.max(x[n, c, h:h + self.length, w:w + self.length])
        self.output_activations.data = data
        return self.output_activations

    def backward(self):
        r"""
        Backward function for maxpool layer.

        In the forward pass we are taking the inputs and subsampling them to use max of a particular block.
        For the derivative of a maxpool function would the zeros of the size of input dimensions masking the maxed
        elements since they are the ones which were activated.
        """
        data = np.zeros(self.input_activations.shape)
        n1, c1, h1, w1 = self.output_activations.shape
        s = self.stride
        l1 = self.length
        for n in range(n1):
            for c in range(c1):
                for h in range(h1):
                    for w in range(w1):
                        x_pool = self.input_activations.data[n, c, h * s:h * s + l1, w * s:w * s + l1]
                        mask = (x_pool == np.max(x_pool))
                        data[n, c, h * s:h * s + l1, w * s:w * s + l1] = mask * self.output_activations[n, c, h, w]
        self.input_activations.gradients = data
