import numpy as np
from numba import jit
from scipy.signal import convolve2d


class Conv2DLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        num_kernels = layer_description['num_kernels']
        kernel_size = layer_description['kernel_size']
        num_channels = layer_description['num_channels']
        total_weights = num_channels * kernel_size ** 2
        weights = np.random.randn(num_kernels, num_channels, kernel_size, kernel_size) / np.sqrt(total_weights)
        bias = np.zeros(shape=[num_kernels, 1])

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.weights = weights
        self.bias = bias

        self.weights_gradients = []
        self.bias_gradients = []

    def forward(self, layer_input):
        self.cache, layer_output = _compiled_forward(layer_input, self.weights, self.bias)
        return layer_output

    def derive(self, dout):
        dx, dw, db = _compiled_derive(self.cache, self.kernel_size, self.weights, dout)

        self.weights_gradients.append(dw)
        self.bias_gradients.append(db)

        return dx

    def update_weights(self, lr):
        self.weights = _compiled_weight_update(self.weights, np.array(self.weights_gradients), lr)
        self.weights_gradients = []

        self.bias = _compiled_weight_update(self.bias, np.array(self.bias_gradients), lr)
        self.bias_gradients = []


@jit(nopython=True)
def _compiled_forward(layer_input, weights, bias):
    pad = 1
    stride = 1
    C, H, W = layer_input.shape
    F, C, FH, FW = weights.shape

    outH = 1 + (H - FH + 2 * pad) / stride
    outW = 1 + (W - FW + 2 * pad) / stride
    outH = int(outH)
    outW = int(outW)

    x_pad = np.zeros((C, H + 2 * pad, W + 2 * pad))
    x_pad[:, pad:-pad, pad:-pad] = layer_input

    H_pad, W_pad = x_pad.shape[1], x_pad.shape[2]

    # create w_row matrix
    w_row = weights.reshape(F, C * FH * FW)  # [F x C*FH*FW]

    x_col = np.zeros((C * FH * FW, outH * outW))  # [C*FH*FW x H'*W']
    neuron = 0
    for i in range(0, H_pad - FH + 1, stride):
        for j in range(0, W_pad - FW + 1, stride):
            x_col[:, neuron] = x_pad[:, i:i + FH, j:j + FW].copy().reshape(C * FH * FW)
            neuron += 1

    layer_output = (w_row.dot(x_col) + bias.reshape(F, 1)).copy().reshape(F, outH, outW)
    cache = x_pad

    return cache, layer_output


@jit(nopython=True)
def _compiled_derive(cache, kernel_size, weights, dout):
    x_pad = cache
    FH = kernel_size
    FW = kernel_size
    F, outH, outW = dout.shape
    C, Hpad, Wpad = x_pad.shape
    stride = 1
    pad = 1

    # create w_row matrix
    w_row = weights.reshape(F, C * FH * FW)  # [F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C * FH * FW, outH * outW))  # [C*FH*FW x H'*W']
    out_col = dout.reshape(F, outH * outW)  # [F x H'*W']
    w_out = w_row.T.dot(out_col)  # [C*FH*FW x H'*W']
    dx_cur = np.zeros((C, Hpad, Wpad))
    neuron = 0
    for i in range(0, Hpad - FH + 1, stride):
        for j in range(0, Wpad - FW + 1, stride):
            dx_cur[:, i:i + FH, j:j + FW] += w_out[:, neuron].copy().reshape(C, FH, FW)
            x_col[:, neuron] = x_pad[:, i:i + FH, j:j + FW].copy().reshape(C * FH * FW)
            neuron += 1
    dx = dx_cur[:, pad:-pad, pad:-pad]
    dw = out_col.dot(x_col.T).reshape(F, C, FH, FW)
    db = out_col.sum(axis=1)
    return dx, dw, db


@jit(nopython=True)
def _compiled_weight_update(weights, weights_gradients, lr):
    weights_gradients = weights_gradients.sum(axis=0).reshape(weights.shape)
    weights = weights - lr * weights_gradients
    return weights


if __name__ == '__main__':
    np.random.seed(69)
    desc = {"type": "conv2d",
            "kernel_size": 5,
            "num_kernels": 4,
            "num_channels": 3}
    l = Conv2DLayer(desc)
    inp = np.random.randn(3, 16, 16)
    import time

    s_t = time.time()
    out = l.forward(inp)

    print(time.time() - s_t)

    s_t = time.time()
    out = l.forward(inp)

    print(time.time() - s_t)
    s_t = time.time()
    out = l.forward(inp)

    print(time.time() - s_t)
    exit()
    #
    # x = l.forward(x)
    # x = l.derive(x)
    #
    # print(x)

    inp = np.random.randn(32, 32, 4)
    s_t = time.time()
    print(time.time() - s_t)
    s_t = time.time()
    l.derive(inp)
    print(time.time() - s_t)
    s_t = time.time()
    l.derive(inp)
    print(time.time() - s_t)
