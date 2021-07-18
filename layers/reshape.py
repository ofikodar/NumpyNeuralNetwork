import numpy as np
from numba import jit


class FlattenLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']

    def forward(self, layer_input,is_train):
        self.cache = layer_input
        batch_size = layer_input.shape[0]
        return layer_input.reshape(batch_size, -1)

    def derive(self, dz):
        return dz.reshape(self.cache.shape)

    def update_weights(self, lr):
        pass


class MaxPoolLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.pool_size = layer_description['pool_size']

    def forward(self, layer_input,is_train):
        self.cache, pool_output = _compiled_forward(layer_input, self.pool_size)
        return pool_output

    def derive(self, dx):
        dx = _compiled_derive(self.cache, dx, self.pool_size)
        return dx

    def update_weights(self, lr):
        pass


# def _compiled_forward(layer_input, pool_size):
    batch_size, channels, image_h, image_w = layer_input.shape
    pool_out_h = int(1 + (image_h - pool_size) / pool_size)
    pool_out_w = int(1 + (image_w - pool_size) / pool_size)

    # create output tensor for pooling layer
    pool_output = np.zeros((batch_size, channels, pool_out_h, pool_out_w))
    for index in range(batch_size):
        out_col = np.zeros((channels, pool_out_h * pool_out_w))
        neuron = 0
        for i in range(0, image_h - pool_size + 1, pool_size):
            for j in range(0, image_w - pool_size + 1, pool_size):
                pool_region = layer_input[index, :, i:i + pool_size, j:j + pool_size].copy().reshape(channels, -1)
                out_col[:, neuron] = pool_region.max(axis=1)
                neuron += 1
        pool_output[index] = out_col.copy().reshape(channels, pool_out_h, pool_out_w)

    cache = layer_input
    return cache, pool_output


# def _compiled_derive(cache, dx, pool_size):
    layer_input = cache
    batch_size, channels, pool_output_h, pool_output_w = dx.shape
    image_h, image_w = layer_input.shape[2], layer_input.shape[3]

    dx_next = np.zeros(layer_input.shape)

    for index in range(batch_size):
        dout_row = dx[index].reshape(channels, -1)
        neuron = 0
        for i in range(0, image_h - pool_size + 1, pool_size):
            for j in range(0, image_w - pool_size + 1, pool_size):
                pool_region = layer_input[index, :, i:i + pool_size, j:j + pool_size].reshape(channels, -1)
                max_pool_indices = pool_region.argmax(axis=1)
                dout_cur = dout_row[:, neuron]
                neuron += 1

                dmax_pool = np.zeros(pool_region.shape)
                dmax_pool[np.arange(channels), max_pool_indices] = dout_cur
                dx_next[index, :, i:i + pool_size, j:j + pool_size] += dmax_pool.reshape(channels, pool_size, pool_size)
    dx = dx_next
    return dx


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "maxPooling",
            "pool_size": 2}
    l = MaxPoolLayer(desc)
    inp = (np.random.randn(1, 1, 8, 8) * 10).astype(int)
    import time
    x = l.forward(inp)
    s_t = time.time()
    zzz = l.derive(x)
    # print(time.time() - s_t)
