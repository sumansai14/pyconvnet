from .conv import ConvLayer
from .inputl import InputLayer
from .fc import FullyConnectedLayer
from .nonlinearites import ReLULayer, SigmoidLayer, TanhLayer


from enum import Enum


class Layers(Enum):
    CONV = ConvLayer
    INPUT = InputLayer
    FC = FullyConnectedLayer


class Activations(Enum):
    RELU = ReLULayer
    SIGMOID = SigmoidLayer
    TANH = TanhLayer
