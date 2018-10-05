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
        r"""
        We're storing gradients of gradients with very vector.

        The problem is the shape of the gradient of a input activations will be different from the shape of the input itself.
        We'll have to tackle this.

        EASY PEASY.
        """
        p = self.padding
        x = self.input_activations
        x_pad = np.lib.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values=0)
        x_gradients = np.zeros(x_pad.shape)
        w_gradients = np.zeros(self.fweights.shape)
        b_gradients = np.zeros(self.fbiases.shape)
        # print(self.input_activations.data.shape)
        # # print(self.input_activations.gradients.shape)
        # print(self.output_activations.data.shape)
        # print(self.output_activations.gradients.shape)
        for n in range(self.input_activations.shape[0]):
            for f in range(self.fshape[0]):
                for h in range(0, x.data.shape[2], self.stride):
                    for w in range(0, x.data.shape[3], self.stride):
                        x_gradients[n, f, h:h + self.fshape[2], w:w + self.fshape[3]] += self.output_activations.gradients[n, f, h, w] * self.fweights.data[f, :, :, :]
        # Delete Padding to match shapes
        delete_height = range(self.padding) + range(x.shape[2] + self.padding, x.shape[2] + (2 * self.padding), 1)
        delete_width = range(self.padding) + range(x.shape[3] + self.padding, x.shape[3] + (2 * self.padding), 1)
        np.delete(x_gradients, delete_height, axis=2)
        np.delete(x_gradients, delete_width, axis=3)
        for n in range(self.input_activations.shape[0]):
            for f in range(self.fshape[0]):
                for h_f in range(0, self.output_activations.gradients.shape[2]):
                    for w_f in range(0, self.output_activations.gradients.shape[3]):
                        w_gradients[n, :, :, :] += self.output_activations.gradients[n, f, h, w] + x_pad[n, :, h_f * self.stride: h_f * self.stride + self.fshape[2], w_f * self.stride:w_f * self.stride + self.fshape[3]]

        for f in range(self.fshape[0]):
            b_gradients[f] = np.sum(self.output_activations.gradients[:, f, :, :])

        self.input_activations.gradients = x_gradients
        self.fweights.gradients = w_gradients
        self.fbiases.gradients = b_gradients
