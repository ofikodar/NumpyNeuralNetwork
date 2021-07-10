import numpy as np
from numba import jit


class Sigmoid:
    def __init__(self, layer_description):
        self.type = layer_description['type']

    def forward(self, layer_input):
        sigmoid_z = _compiled_forward_sigmoid(layer_input)
        self.cache = sigmoid_z
        return sigmoid_z

    def derive(self, dz):
        return _compiled_derive_sigmoid(dz, self.cache)

    def update_weights(self, lr):
        pass


@jit(nopython=True)
def _compiled_forward_sigmoid(layer_input):

    output = 1 / (1 + np.exp(-layer_input))
    return output


@jit(nopython=True)
def _compiled_derive_sigmoid(dz, sigmoid_z):
    d_sigmoid = sigmoid_z * (1 - sigmoid_z)
    output = dz * d_sigmoid

    return output


class LeakyRelu:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.alpha = 0.1

    def forward(self, layer_input):
        z = np.maximum(0, layer_input)

        self.cache = layer_input

        return z

    def derive(self, dz):
        layer_input = self.cache
        dz = dz * (layer_input > 0)

        return dz

    def update_weights(self, lr):
        pass


class Softmax:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.cache = []

    def forward(self, layer_input):
        self.cache = layer_input
        output = _compiled_forward_softmax(layer_input)
        return output

    def derive(self, a):
        dz = self.cache - a
        return dz

    def update_weights(self, lr):
        pass


@jit(nopython=True)
def _compiled_forward_softmax(layer_input):
    e_x = np.exp(layer_input - np.max(layer_input))
    output = e_x / e_x.sum(axis=0)
    return output


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "softmax"}
    l = Softmax(desc)
    inp = np.random.randn(64).reshape(-1, 1)

    import time

    s_t = time.time()
    x = l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    x = l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    x = l.forward(inp)
    print(time.time() - s_t)
    print(x.sum())
