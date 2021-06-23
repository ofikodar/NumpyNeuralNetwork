import numpy as np
from numba import jit, njit
from scipy.signal import convolve2d
import time


class Conv2DLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        num_kernels = layer_description['num_kernels']
        kernel_size = layer_description['kernel_size']
        num_channels = layer_description['num_channels']
        total_weights = num_channels * kernel_size ** 2
        weights = np.random.randn(num_kernels, kernel_size, kernel_size, num_channels) / np.sqrt(total_weights)
        bias = np.zeros(shape=[num_kernels, 1])

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.weights = weights
        self.bias = bias

        self.weights_gradients = []
        self.bias_gradients = []

    def forward(self, layer_input):
        self.cache = layer_input
        h, w, channel_size = layer_input.shape

        layer_output = np.zeros([h, w, self.num_kernels])
        for _filter_index in range(self.num_kernels):
            for _channel_index in range(channel_size):
                layer_output[:, :, _filter_index] += convolve2d(layer_input[:, :, _channel_index],
                                                                self.weights[_filter_index, _channel_index],
                                                                mode='same')
        return layer_output

    def derive(self, dz):
        da, dw, db = _compiled_derive(self.cache, self.kernel_size, self.weights, dz)

        self.weights_gradients.append(dw)
        self.bias_gradients.append(db)

        return da

    def derive_2(self, dz):

        a_prev = self.cache
        prev_h, prev_w, prev_c = a_prev.shape

        out_h, out_w, out_c = dz.shape
        dw = np.zeros((out_c, self.kernel_size, self.kernel_size, prev_c))
        db = np.zeros(out_c)

        pad_size = int((self.kernel_size - 1) / 2)
        a_prev_pad = np.zeros((prev_h + 2 * pad_size, prev_w + 2 * pad_size, prev_c))
        a_prev_pad[pad_size:-pad_size, pad_size:-pad_size] = a_prev
        da_prev_pad = np.zeros(a_prev_pad.shape)

        for h in range(out_h):
            for w in range(out_w):
                for c in range(out_c):
                    slice = a_prev_pad[h:h + self.kernel_size, w:w + self.kernel_size, :]
                    da_prev_pad[h:h + self.kernel_size, w:w + self.kernel_size, :] += self.weights[c] * dz[h, w, c]

                    dw[c] += slice * dz[h, w, c]
                    db[c] += dz[h, w, c]

        da = da_prev_pad[pad_size:-pad_size, pad_size:-pad_size, :]
        self.weights_gradients.append(dw)
        self.bias_gradients.append(db)

        return da

    def update_weights(self, lr):
        # self.weights_gradients = self.weights_gradients[1:]
        # self.bias_gradients = self.bias_gradients[1:]

        self.weights_gradients = np.array(self.weights_gradients).sum(axis=0)

        self.weights = self.weights - lr * self.weights_gradients

        self.weights_gradients = []

        self.bias_gradients = np.array(self.bias_gradients).sum(axis=0)

        self.bias = self.bias - lr * self.bias_gradients
        self.bias_gradients = []


@njit
def _compiled_derive(cache, kernel_size, weights, dz):
    a_prev = cache
    prev_h, prev_w, prev_c = a_prev.shape

    out_h, out_w, out_c = dz.shape
    dw = np.zeros((out_c, kernel_size, kernel_size, prev_c))
    db = np.zeros(out_c)

    pad_size = int((kernel_size - 1) / 2)
    a_prev_pad = np.zeros((prev_h + 2 * pad_size, prev_w + 2 * pad_size, prev_c))
    a_prev_pad[pad_size:-pad_size, pad_size:-pad_size] = a_prev
    da_prev_pad = np.zeros(a_prev_pad.shape)

    for h in range(out_h):
        for w in range(out_w):
            for c in range(out_c):
                slice = a_prev_pad[h:h + kernel_size, w:w + kernel_size, :]
                da_prev_pad[h:h + kernel_size, w:w + kernel_size, :] += weights[c] * dz[h, w, c]

                dw[c] += slice * dz[h, w, c]
                db[c] += dz[h, w, c]

    da = da_prev_pad[pad_size:-pad_size, pad_size:-pad_size, :]
    return da, dw, db


if __name__ == '__main__':
    np.random.seed(69)
    desc = {"type": "conv2d",
            "kernel_size": 3,
            "num_kernels": 4,
            "num_channels": 3}
    l = Conv2DLayer(desc)
    inp = np.random.randn(32, 32, 3)
    import time

    x = l.forward(inp)
    inp = np.random.randn(32, 32, 4)
    s_t = time.time()
    l.derive(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.derive(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.derive(inp)
    print(time.time() - s_t)