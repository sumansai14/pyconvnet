import requests
import os
import numpy as np
import struct
from collections import defaultdict as ddict
import gzip
# import ipdb as pdb


class Dataset(object):
    """
    Every dataset which subclasses this gives the test, train and valid data in pairwise (features, labels) tuple.

    Need to implement the following functions:
    1. an iterator on top of entire data
    2. a batch iterator which has the capability to shuffle

    """

    train = None
    test = None
    valid = None

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    def batches(self, batch_size, axis=0, shuffle=False):
        self.randomize()
        n = self.train[0].shape[axis]
        for k in range(0, n, batch_size):
            yield (
                self.train[0].take(indices=range(k, min(k + batch_size, n)), axis=axis),
                self.train[1].take(indices=range(k, min(k + batch_size, n)), axis=axis)
            )

    def randomize(self):
        raise NotImplementedError()


class MNISTDataSet(Dataset):
    # ideally we want to make this cache path that is configurable and download data there.
    absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = 'data/mnist/'
    urls = [
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    @property
    def data(self):
        file_name = self.urls[0].split('/')[-1]
        print(f"file name {file_name}")
        if not os.path.exists(os.path.join(self.absolute_path, self.data_path, file_name)):
            print("Data is not present, Downloading now...")
            self.fetch_data()
        return self.load_data()

    def load_data(self):
        if not os.path.exists(os.path.join(self.absolute_path, self.data_path)):
            self.fetch_data()
        if len(os.listdir(os.path.join(self.absolute_path, self.data_path))) != 4:
            raise IOError("dataset not found")
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
            # Get paths of the images and labels
            dataset = ddict(dict)
            dataset['train']['img_path'] = os.path.join(self.absolute_path, self.data_path, 'train-images-idx3-ubyte.gz')
            dataset['train']['lbl_path'] = os.path.join(self.absolute_path, self.data_path, 'train-labels-idx1-ubyte.gz')
            dataset['test']['img_path'] = os.path.join(self.absolute_path, self.data_path, 't10k-images-idx3-ubyte.gz')
            dataset['test']['lbl_path'] = os.path.join(self.absolute_path, self.data_path, 't10k-labels-idx1-ubyte.gz')

            for split in ['train', 'test']:
                with gzip.open(dataset[split]['lbl_path'], 'rb') as f:
                    magic, num = struct.unpack(">II", f.read(8))
                    dataset[split]['labels'] = np.frombuffer(f.read(), dtype=np.int8)
                    # dimension normalization
                    dataset[split]['labels'] = dataset[split]['labels'].reshape(dataset[split]['labels'].shape[0], 1)
                with gzip.open(dataset[split]['img_path'], 'rb') as f:
                    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                    dataset[split]['images'] = np.frombuffer(f.read(), dtype=np.uint8).reshape(len(dataset[split]['labels']), rows, cols)

            self.train = (dataset['train']['images'][:50000], dataset['train']['labels'][:50000])
            self.valid = (dataset['train']['images'][50000:], dataset['train']['labels'][50000:])
            self.test = (dataset['test']['images'], dataset['test']['labels'])
            return

    def fetch_data(self, force=False):
        if not os.path.exists(os.path.join(self.absolute_path, self.data_path)):
            os.makedirs(os.path.join(self.absolute_path, self.data_path))
        for url in self.urls:
            file_name = url.split('/')[-1]
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(os.path.join(self.absolute_path, self.data_path, file_name), 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            else:
                raise ("Failed to download the dataset")

    def randomize(self, axis=0):
        permuation = np.random.permutation(self.train[0].shape[axis])
        self.train = (self.train[0].take(indices=permuation, axis=axis), self.train[1].take(indices=permuation, axis=axis))
        return
