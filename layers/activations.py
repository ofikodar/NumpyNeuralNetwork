from scipy.special._logsumexp import softmax
import numpy as np

class Sigmoid:
    def __init__(self, layer_description):
        self.type = layer_description['type']

    def forward(self, layer_input):
        self.cache = layer_input

        output = 1 / (1 + np.exp(-layer_input))
        return output

    def derive(self, dz):
        z = self.cache
        sigmoid_z = self.forward(z.copy())
        d_sigmoid = sigmoid_z * (1 - sigmoid_z)
        return dz * d_sigmoid

    def update_weights(self, lr):
        pass


class LeakyRelu:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.alpha = 0.1

    def forward(self, layer_input):
        self.cache = layer_input

        z = layer_input.copy()
        is_neg = z < 0
        z[is_neg] = self.alpha * z[is_neg]
        return z

    def derive(self, dz):
        z = self.cache
        derivative = np.zeros(shape=z.shape)
        is_neg = z < 0

        derivative[is_neg] = self.alpha
        derivative[~is_neg] = 1

        return dz * derivative

    def update_weights(self, lr):
        pass


class Softmax:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.cache = []

    def forward(self, layer_input):
        self.cache = layer_input

        output = softmax(layer_input)

        return output

    def derive(self, a):
        dz = self.cache - a
        return dz

    def update_weights(self, lr):
        pass
