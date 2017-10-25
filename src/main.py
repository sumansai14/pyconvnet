from .network import Network
from .datasets import MNISTDataSet
from .layers import Layers

train_x, train_y, validation_data, test_x, test_y = MNISTDataSet().data
layers = [
    {'type': Layers.INPUT, 'input_len': train_x.shape[0]},
    {'type': Layers.FC, 'input_len': train_x.shape[0], 'num_neurons': 30}
]

network = Network()
