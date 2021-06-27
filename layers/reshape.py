import numpy as np
from numba import jit


class FlattenLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']

    def forward(self, layer_input):
        self.cache = layer_input
        return layer_input.reshape(-1, 1)

    def derive(self, dz):
        return dz.reshape(self.cache.shape)

    def update_weights(self, lr):
        pass


class MaxPoolLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        self.pool_size = layer_description['pool_size']

    def forward(self, layer_input):
        self.cache, pool_output = _compiled_forward(layer_input, self.pool_size)
        return pool_output

    def derive(self, dz):
        return self.cache

    def update_weights(self, lr):
        pass


@jit(nopython=True)
def _compiled_forward(layer_input, pool_size):
    rows, cols, channels = layer_input.shape
    _rows_steps = rows // pool_size
    _cols_steps = cols // pool_size

    cache = np.zeros(layer_input.shape)
    pool_output = np.zeros((_rows_steps, _cols_steps, channels))
    for _r in range(0, rows, _rows_steps):
        for _c in range(0, cols, _cols_steps):
            for _channel_idx in range(channels):
                window = layer_input[_r:_r + pool_size, _c:_c + pool_size, _channel_idx]
                window_max = window.max()
                rows_pool_index = _r // pool_size
                cols_pool_index = _c // pool_size
                pool_output[rows_pool_index, cols_pool_index, _channel_idx] = window_max

                window_argmax = window.argmax()
                window_row_index = window_argmax // pool_size
                window_col_index = window_argmax % pool_size

                cache[_r + window_row_index, _c + window_col_index, _channel_idx] = 1

    return cache, pool_output


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "maxPooling",
            "pool_size": 2,
            "num_kernels": 4,
            "num_channels": 3}
    l = MaxPoolLayer(desc)
    inp = np.random.randn(32, 32, 3)
    import time

    s_t = time.time()
    x = l.forward(inp)
    print(time.time() - s_t)
    s_t = time.time()
    x = l.forward(inp)
    print(time.time() - s_t)
