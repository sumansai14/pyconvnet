from .conv import ConvLayer
from .inputl import InputLayer
from .fc import FullyConnectedLayer
from .pool import MaxPool
from .nonlinearities import SigmoidLayer, ReLULayer
from .flatten import FlattenLayer


from enum import Enum


class Layers(Enum):
    CONV = ConvLayer
    INPUT = InputLayer
    FC = FullyConnectedLayer
    MAXPOOL = MaxPool
    FLATTEN = FlattenLayer


class Activations(Enum):
    RELU = ReLULayer
    SIGMOID = SigmoidLayer
    # TANH = TanhLayer
