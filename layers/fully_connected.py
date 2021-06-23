import numpy as np


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
        self.cache = []

    def forward(self, layer_input):
        self.cache = layer_input
        z = np.dot(self.weights, layer_input) + self.bias
        return z

    def derive(self, dz):
        dw = np.dot(dz, self.cache.T)
        db = dz
        self.weights_gradients.append(dw)
        self.bias_gradients.append(db)
        dh_prev = np.dot(self.weights.T, dz)
        return dh_prev

    def update_weights(self, lr):
        self.weights_gradients = np.array(self.weights_gradients).sum(axis=0)
        self.weights = self.weights - lr * self.weights_gradients
        self.weights_gradients = []

        self.bias_gradients = np.array(self.bias_gradients).sum(axis=0)
        self.bias = self.bias - lr * self.bias_gradients
        self.bias_gradients = []
