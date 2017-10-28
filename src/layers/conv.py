from .baselayer import Layer
from .vector import Vector
import numpy as np


class ConvLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self.stride = kwargs['stride']
        self.filters = kwargs['filters']
        self.padding = kwargs.get('padding', 0)
        self.input_activations = None()
        self.output_activations = Vector()
        # Considering all the images are sqaures for now.
        
