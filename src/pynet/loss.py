r"""For now, we implement only a single loss per network. We'll take care of multi task learning later."""
from pynet.layers.vector import Vector
import numpy as np


class MSELoss(object):

    def __init__(self):
        self.input_activations = None
        self.output_activations = Vector()

    def forward(self, x, y):
        self.input_activations = x
        self.output_activations.data = np.mean((x.data - y).dot((x.data - y).T)) / 2.0
        accuracy = np.mean(np.argmax(x.data, axis=0) == np.argmax(y, axis=0))
        return accuracy, self.output_activations.data

    def backward(self, y):
        self.input_activations.gradients = (self.input_activations.data - y)
        return
