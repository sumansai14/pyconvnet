import numpy as np
import logging
# from layers.v

class Estimator(object):
    """
    This is losely based on the tensorflow estimator:

    We'll first implement the MNIST estimator and bring the common functionality here.

    Now that i think about it, data transformation can be a bitch - Need to think through this thoroughly. But regarding the API's that we expose,
    I think we can just take that up from tensorflows implementation.

    1. takes a function that transforms the input dataset into a format that the network can consume.
    """
    dataset = None
    optimizer = None
    network = None
    transformer = None

    def __init__(self, dataset, optimizer, network, transformer):
        self.dataset = dataset
        self.optimizer = optimizer
        self.network = network
        # transformer is a function that gives batches fo train features, labels.
        self.transformer = transformer

        self.logger = logging.getLogger(__name__)

    def train(self, epochs):
        for epoch in range(epochs):
            mini_batches = self.dataset.batches(self.optimizer.batch_size)
            train_loss = 0
            train_acc = 0
            for step, batch in enumerate(mini_batches):
                x, y = self.transformer(*batch)
                output = self.network.forward(x)  # Predict
                accuracy, loss = self.network.loss(output, y)  # Cal'c Loss
                train_acc += accuracy
                train_loss += loss
                self.network.backward(output, y)  # Backword Prop
                self.optimizer.step()  # Update Gradients
                self.optimizer.zero_grad()  # Zero gradients
            # Here we have to accumulate the training accuracy across all the steps.
            train_acc /= step
            train_loss /= step
            self.logger.info('TRAIN: Completed epoch {epoch} with accuracy {accuracy}% and loss {loss.data}'.format(epoch=epoch, accuracy=(train_acc * 100), loss=train_loss))
            if self.dataset.valid:
                valid = self.dataset.valid
                valid_x, valid_y = self.transformer(*valid)
                valid_acc, valid_loss = self.network.loss(self.network.forward(valid_x), valid_y)
            self.logger.info('VALID: Completed epoch for {epoch} with accuracy {accuracy}% and loss {loss.data}'.format(epoch=epoch, accuracy=(valid_acc * 100), loss=valid_loss))

    def test(self):
        test = self.dataset.test
        x, y = self.transformer(*test)
        output = self.network.forward(x)  # Predict
        accuracy, loss = self.network.loss(output, y)  # Cal'c Loss
        self.logger.info('TEST: Model trained with test accuracy {accuracy}% and loss {loss.data}'.format(accuracy=(accuracy * 100), loss=loss))
