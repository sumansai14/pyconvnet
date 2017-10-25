from .layers import ConvLayer, SoftMax, FC

class Network(object):

    def __init__(self, layers, optimizers, loss):
        