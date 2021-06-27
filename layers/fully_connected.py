import numpy as np
from numba import jit


class FCLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        num_nodes = layer_description['num_nodes']
        prv_layer_shape = layer_description['previous_nodes']

        weights = np.random.randn(num_nodes, prv_layer_shape) / np.sqrt(num_nodes)
        bias = np.zeros(shape=[num_nodes, 1])
        self.weights = weights
        self.bias = bias
        self.weights_gradients = []
        self.bias_gradients = []

    def forward(self, layer_input):
        self.cache = layer_input
        z = _compiled_forward(layer_input, self.weights, self.bias)

        return z

    def derive(self, dz):
        dh_prev, dw, db = _compiled_derive(self.cache, dz, self.weights)
        self.weights_gradients.append(dw)
        self.bias_gradients.append(db)
        return dh_prev

    def update_weights(self, lr):
        self.weights = _compiled_weight_update(self.weights, np.array(self.weights_gradients), lr)
        self.weights_gradients = []

        self.bias = _compiled_weight_update(self.bias, np.array(self.bias_gradients), lr)
        self.bias_gradients = []


@jit(nopython=True)
def _compiled_forward(layer_input, weights, bias):
    z = np.dot(weights, layer_input) + bias
    return z


@jit(nopython=True)
def _compiled_derive(cache, dz, weights):
    dw = np.dot(dz, cache.T)
    db = dz
    dh_prev = np.dot(weights.T, dz)
    return dh_prev, dw, db


@jit(nopython=True)
def _compiled_weight_update(weights, weights_gradients, lr):
    weights_gradients = weights_gradients.sum(axis=0)
    weights = weights - lr * weights_gradients
    return weights


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "fc",
            "num_nodes": 10,
            "previous_nodes": 64}
    l = FCLayer(desc)
    inp = np.random.randn(64).reshape(-1, 1)

    import time

    x = l.forward(inp)
    _ = l.derive(x)
    s_t = time.time()
    _ = l.update_weights(0.01)

    print(time.time() - s_t)
    s_t = time.time()
    _ = l.update_weights(0.01)
    print(time.time() - s_t)
    s_t = time.time()
    _ = l.update_weights(0.01)
    print(time.time() - s_t)
