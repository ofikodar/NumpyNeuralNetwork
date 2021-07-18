import numpy as np
from numba import jit, njit
from scipy.signal import convolve2d


class Conv2DLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        output_channels = layer_description['output_channels']
        kernel_size = layer_description['kernel_size']
        input_channels = layer_description['input_channels']
        total_weights = input_channels * kernel_size ** 2
        weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) / np.sqrt(total_weights)
        bias = np.zeros(shape=[output_channels])

        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.weights = weights
        self.bias = bias
        self.cache = None
        self.weights_gradients = []
        self.bias_gradients = []

    def forward(self, layer_input, is_train):
        self.cache, layer_output = _compiled_forward(layer_input, self.weights, self.bias)
        return layer_output

    def derive(self, dx):
        dx, dw, db = _compiled_derive(self.cache, self.weights, dx)

        self.weights_gradients = dw
        self.bias_gradients = db

        return dx

    def update_weights(self, lr):
        self.weights = self.weights - lr * self.weights_gradients
        self.bias = self.bias - lr * self.bias_gradients

@njit()
def _compiled_forward(layer_input, weights, bias):
    batch_size, input_channels, image_h, image_w = layer_input.shape
    output_channels = weights.shape[0]
    kernel_size = weights.shape[-1]
    output_image = np.zeros((batch_size, output_channels, image_h, image_w))

    pad = int((kernel_size - 1 / 2))
    half_pad = int(pad / 2)
    x_pad = np.zeros((batch_size, input_channels, image_h + pad, image_w + pad))
    x_pad[:, :, half_pad:-half_pad, half_pad:-half_pad] = layer_input
    image_h_pad, image_w_pad = x_pad.shape[2], x_pad.shape[3]
    w_row = weights.reshape(output_channels, -1)

    x_col = np.zeros((w_row.shape[1], image_h * image_w))
    for index in range(batch_size):
        neuron = 0
        for i in range(0, image_h_pad - kernel_size + 1):
            for j in range(0, image_w_pad - kernel_size + 1):
                flatten = x_pad[index, :, i:i + kernel_size, j:j + kernel_size].copy().reshape(-1)
                x_col[:, neuron] = flatten
                neuron += 1
        output_image[index] = (w_row.dot(x_col) + bias.reshape(output_channels, 1)).reshape(output_channels,
                                                                                                   image_h, image_w)
    cache = x_pad
    return cache, output_image.reshape(batch_size, output_channels, image_h, image_w)

def _compiled_derive(cache, weights, dx):
    x_pad = cache
    image_h_pad = x_pad.shape[2]
    image_w_pad = x_pad.shape[3]

    batch_size, out_channels, image_h, image_w = dx.shape
    input_channels = x_pad.shape[1]
    kernel_size = weights.shape[-1]
    pad = int((kernel_size - 1 / 2))
    half_pad = int(pad / 2)

    dx_next = np.zeros((batch_size, input_channels, image_h, image_w))
    dw, db = np.zeros(weights.shape), np.zeros(out_channels)

    w_row = weights.reshape(out_channels, -1)
    x_col = np.zeros((w_row.shape[1], image_h * image_w))
    for index in range(batch_size):
        out_col = dx[index].reshape(out_channels, -1)
        w_out = w_row.T.dot(out_col)
        dx_cur = np.zeros((input_channels, image_h_pad, image_w_pad))
        neuron = 0
        for i in range(0, image_h_pad - kernel_size + 1):
            for j in range(0, image_w_pad - kernel_size + 1):
                dx_cur[:, i:i + kernel_size, j:j + kernel_size] += w_out[:, neuron].reshape(input_channels, kernel_size,
                                                                                            kernel_size)
                x_col[:, neuron] = x_pad[index, :, i:i + kernel_size, j:j + kernel_size].reshape(-1)
                neuron += 1
        dx_next[index] = dx_cur[:, half_pad:-half_pad, half_pad:-half_pad]
        dw += out_col.dot(x_col.T).reshape(out_channels, input_channels, kernel_size, kernel_size)
        db += out_col.sum(axis=1)

    dx = dx_next
    return dx, dw, db


if __name__ == '__main__':
    np.random.seed(69)
    desc = {"type": "conv2d",
            "kernel_size": 3,
            "input_channels": 128,
            "output_channels": 256}
    l = Conv2DLayer(desc)
    inp = np.random.randn(32, 128, 32, 32)
    import time

    out = l.forward(inp,is_train=False)

    s_t = time.time()
    _ = l.derive(out)
    e_t = time.time()
    print(e_t - s_t)
    s_t = time.time()
    _ = l.derive(out)
    e_t = time.time()
    print(e_t - s_t)
    _ = l.derive(out)
    e_t = time.time()
    print(e_t - s_t)