from .baselayer import Layer
from .vector import Vector
import numpy as np


class ConvLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self.stride = kwargs['stride']
        # we'll be passing padding to keep spatial info constant or we can calculate this later.
        self.padding = kwargs.get('padding', 1)
        self.fshape = kwargs.get('fshape')  # format (num_filters, channels, height, width)

        self.input_activations = None
        self.output_activations = Vector()
        n_out = (self.fshape[0] * np.prod(self.fshape[2:]) / np.prod(self.fshape[0]))
        self.fweights = Vector(data=np.random.normal(loc=0, scale=np.sqrt(1 / n_out), size=self.fshape))
        self.fbiases = Vector(data=np.random.normal(loc=0, scale=1.0, size=self.fshape[0]))
        # Considering all the images are sqaures for now.

    def forward(self, x, is_training):
        r"""
        Let the convolutions begin.

        :param x: input images of dimensions(num_images, channels, height, width)
        """
        self.input_activations = x
        height = 1 + ((x.shape[2] + (2 * self.padding) - self.fshape[2]) / self.stride)
        width = 1 + ((x.shape[3] + (2 * self.padding) - self.fshape[3]) / self.stride)
        output_shape = (x.shape[0], self.fshape[0], int(height), int(width))
        print(output_shape)
        data = np.zeros(output_shape)
        p = self.padding
        x_pad = np.lib.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values=0)
        for n in range(x.data.shape[0]):
            for f in range(self.fshape[0]):
                for h in range(0, x.data.shape[2], self.stride):
                    for w in range(0, x.data.shape[3], self.stride):
                        data[n, f, int(h / self.stride), int(w / self.stride)] = np.sum(x_pad[n, :, h:(h + self.fshape[2]), w:(w + self.fshape[3])] * self.fweights.data[f, :, :, :]) + self.fbiases.data[f]

        self.output_activations.data = data
        return self.output_activations

    def backward(self):
        print("nothing backward here")
