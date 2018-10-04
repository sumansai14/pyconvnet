import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(lineno)d:%(name)s] - %(message)s'
)


class MiniBatchGradientDescent(object):
    def __init__(self, network, **kwargs):
        self.network = network
        self.logger = logging.getLogger(__name__)

    def train(self, x, y, mini_batch_size, epochs, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        for epoch in range(epochs):
            x, y = self.randomize(x, y)
            mini_batches = self.get_mini_batces(x, y, mini_batch_size)
            for x, y in mini_batches:
                a = self.network.forward(x)
                accuracy, loss = self.network.backward(a, y)
                self.update_params()  # Since layers don't need to have access to batch size and learning rate
                self.step()
            self.logger.info('completed epoch {epoch} with accuracy {accuracy}% and loss {loss.data}'.format(epoch=epoch, accuracy=(accuracy * 100), loss=loss))

    def randomize(self, x, y):
        permuation = np.random.permutation(x.shape[1])
        x = x[:, permuation]
        y = y[:, permuation]
        return x, y

    def get_mini_batces(self, x, y, mini_batch_size):
        n = x.shape[1]
        mini_batches = [(x[:, k:k + mini_batch_size], y[:, k:k + mini_batch_size]) for k in range(0, n, mini_batch_size)]
        return mini_batches

    def update_params(self):
        params = self.network.get_params_and_grads()
        for vector in params:
            vector.data = vector.data - (self.learning_rate / self.mini_batch_size) * vector.gradients

    def step(self):
        params = self.network.get_params_and_grads()
        for vector in params:
            vector.gradients = None
