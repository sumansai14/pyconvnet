import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(lineno)d:%(name)s] - %(message)s'
)


class MiniBatchGradientDescent(object):
    def __init__(self, network, **kwargs):
        self.network = network
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']

    def step(self):
        params = self.network.get_params_and_grads()
        for vector in params:
            vector.data = vector.data - (self.learning_rate / self.batch_size) * vector.gradients

    def zero_grad(self):
        params = self.network.get_params_and_grads()
        for vector in params:
            vector.gradients = None
