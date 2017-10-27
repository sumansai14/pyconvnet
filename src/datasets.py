import requests
import os
import numpy as np
from struct import unpack
import gzip
import pickle


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class MNISTDataSet(object):
    path = '../data/mnist/'
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    @property
    def data(self):
        file_name = self.urls[0].split('/')[-1]
        if not os.path.exists(self.path + file_name):
            print("Data is not present, Downloading now...")
            self.fetch_data()
        return self.load_data()

    def wrap_data(self, tr_d, va_d, te_d):
        train_x = np.array([np.reshape(x, (784, 1)) for x in tr_d[0]])
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
        train_y = np.array([vectorized_result(y) for y in tr_d[1]])
        train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_x = np.array([np.reshape(x, (784, 1)) for x in te_d[0]])
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        test_y = np.array([vectorized_result(y) for y in te_d[1]])
        test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
        return (train_x, train_y, validation_data, test_x, test_y)

    def load_data(self):
        if len(os.listdir(self.path)) == 4:
            pass
        else:
            """Return the MNIST data as a tuple containing the training data,
            the validation data, and the test data.

            The ``training_data`` is returned as a tuple with two entries.
            The first entry contains the actual training images.  This is a
            numpy ndarray with 50,000 entries.  Each entry is, in turn, a
            numpy ndarray with 784 values, representing the 28 * 28 = 784
            pixels in a single MNIST image.

            The second entry in the ``training_data`` tuple is a numpy ndarray
            containing 50,000 entries.  Those entries are just the digit
            values (0...9) for the corresponding images contained in the first
            entry of the tuple.

            The ``validation_data`` and ``test_data`` are similar, except
            each contains only 10,000 images.

            This is a nice data format, but for use in neural networks it's
            helpful to modify the format of the ``training_data`` a little.
            That's done in the wrapper function ``load_data_wrapper()``, see
            below.
            """
            f = gzip.open(self.path + 'mnist.pkl.gz', 'rb')
            training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
            f.close()
            return self.wrap_data(training_data, validation_data, test_data)

    def fetch_data(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for url in self.urls:
            file_name = url.split('/')[-1]
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(self.path + file_name, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
