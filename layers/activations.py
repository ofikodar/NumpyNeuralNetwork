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


def _compiled_forward_sigmoid(layer_input):
    output = 1 / (1 + np.exp(-layer_input))
    return output


def _compiled_derive_sigmoid(dz, sigmoid_z):
    d_sigmoid = sigmoid_z * (1 - sigmoid_z)
    output = dz * d_sigmoid

    return output


class LeakyRelu:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.alpha = 0.1

    def forward(self, layer_input, is_train):
        z = np.maximum(0, layer_input)
        self.cache = layer_input
        return z

    def derive(self, dx):
        layer_input = self.cache
        dx = dx * (layer_input > 0)
        return dx

    def update_weights(self, lr):
        pass


class Softmax:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.cache = None

    def forward(self, layer_input, is_train):
        output = _compiled_forward_softmax(layer_input)
        self.cache = output
        return output

    def derive(self, y):
        dx = self.cache.copy()
        batch_size = y.shape[0]
        idx = np.arange(batch_size)
        dx[idx, y] -= 1
        dx /= batch_size
        return dx

    def update_weights(self, lr):
        pass


# # def _compiled_forward_softmax(layer_input):
    sub_max = layer_input - np.max(layer_input, axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(sub_max), axis=1, keepdims=True)
    output = np.exp(sub_max - np.log(sum_exp))
    return output


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "Softmax"}
    l = Softmax(desc)
    inp = np.random.randn(128, 10)
    import time

    s_t = time.time()
    l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.forward(inp)
    print(time.time() - s_t)
