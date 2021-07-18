import numpy as np


class BNLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.input_channels = layer_description['input_channels']
        num_nodes = self.input_channels

        self.gamma = np.random.randn(num_nodes) / np.sqrt(num_nodes)
        self.beta = np.random.randn(num_nodes) / np.sqrt(num_nodes)
        self.gamma_gradients = []
        self.beta_gradients = []
        self.cache = []
        self.std = []
        self.running_mean = 0
        self.running_var = 0

    def forward(self, layer_input, is_train=False):

        batch_size, channels, input_h, input_w = layer_input.shape
        layer_input = layer_input.copy().transpose(0, 2, 3, 1).reshape(-1, self.input_channels)
        forward_input = [layer_input, self.gamma, self.beta, self.running_mean, self.running_var, is_train]
        layer_output, self.running_mean, self.running_var, self.cache, self.gamma, self.std = _compiled_forward(
            *forward_input)
        layer_output = layer_output.reshape(batch_size, input_h, input_w, self.input_channels).transpose(0, 3, 1, 2)

        return layer_output

    def derive(self, dx):
        batch_size, channels, input_h, input_w = dx.shape
        dx = dx.copy().transpose(0, 2, 3, 1).reshape(-1, self.input_channels)
        dx, self.gamma_gradients, self.beta_gradients = _compiled_derive(self.cache, self.gamma, self.std, dx)
        dx = dx.reshape(batch_size, input_h, input_w, self.input_channels).transpose(0, 3, 1, 2)

        return dx

    def update_weights(self, lr):
        self.gamma_gradients = self.gamma - lr * self.gamma_gradients
        self.gamma_gradients = self.beta - lr * self.beta_gradients


def _compiled_forward(layer_input, gamma, beta, running_mean, running_var, is_train):
    eps = 1e-5
    momentum = 0.9
    cache = layer_input
    std = None
    if is_train:
        mu = layer_input.mean(axis=0)
        var = layer_input.var(axis=0) + eps
        std = np.sqrt(var)
        z = (layer_input - mu) / std
        layer_output = gamma * z + beta
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * (std ** 2)
        cache = z
    else:
        layer_output = gamma * (layer_input - running_mean) / np.sqrt(running_var + eps) + beta

    return layer_output, running_mean, running_var, cache, gamma, std


def _compiled_derive(cache, gamma, std, dx):
    db = dx.sum(axis=0)
    dg = np.sum(dx * cache, axis=0)

    batch_size = dx.shape[0]
    dz = dx * gamma
    dz_sum = np.sum(dz, axis=0)
    dz_mean = dz_sum / batch_size
    dz_sub = dz_sum - dz_mean
    dx_next = dz_sub - np.sum(dz * cache, axis=0) * cache / batch_size
    dx_next /= std
    dx = dx_next
    return dx, dg, db


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "fc",
            "input_channels": 3}
    l = BNLayer(desc)
    inp = np.random.randn(1, 3, 16, 16) * 100

    import time

    print(inp)
    x = l.forward(inp, is_train=True)

    # l.derive(np.random.randn(1, 64))
    # print(x)
