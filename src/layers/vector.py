r"""Class Definition for Vector."""


class Vector(object):
    r"""
    Class Definition for vector which when instantitated, can be used to stored variables with their gradients.

    The advantage of storing objects in such a manner is, these objects can be passed by reference
    to the previous and next layers where they can be used inside forward and backward propagation.
    """

    def __init__(self, data=None):
        r"""
        We can intialize the vector objects with either some data.

        (in case of layer attirbutes like weigths, biases etc;) or an empty object - in case of acitvations.

        :param data(optional): used to set the data for the vector object.
        returns None
        """
        self.data = data
        self.gradients = None

    @property
    def shape(self):
        r"""
        The shape of the data and gradient are equal and the value is equal to the shape of the vector.

        returns the shape of the vector.
        """
        return self.data.shape

    @property
    def T(self):
        return self.data.T

    def __mul__(self, other):
        return self.data * other
