class Vector(object):

    def __init__(self, data=None):
        self.data = data
        self.gradients = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.data.T

    def __mul__(self, other):
        return self.data * other
