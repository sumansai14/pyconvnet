from .baselayer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.params = kwargs
