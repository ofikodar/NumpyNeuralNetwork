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
        dz = _compiled_derive(self.cache, dz, self.pool_size)
        return dz

    def update_weights(self, lr):
        pass


@jit(nopython=True)
def _compiled_forward(layer_input, pool_size):
    C, H, W = layer_input.shape
    PH = pool_size
    PW = pool_size
    stride = 2
    outH = int(1 + (H - PH) / stride)
    outW = int(1 + (W - PW) / stride)

    pool_output = np.zeros((C, outH * outW))
    neuron = 0
    for i in range(0, H - PH + 1, stride):
        for j in range(0, W - PW + 1, stride):
            pool_region = layer_input[:, i:i + PH, j:j + PW].copy().reshape(C, PH * PW)
            pool_output[:, neuron] = pool_region.max()
            neuron += 1
    pool_output = pool_output.reshape(C, outH, outW)

    cache = layer_input
    return cache, pool_output


@jit(nopython=True)
def _compiled_derive(cache, dout, pool_size):

    x = cache
    C, outH, outW = dout.shape
    H, W = x.shape[1], x.shape[2]
    PH = pool_size
    PW = pool_size
    stride = 2
    # initialize gradient
    dx = np.zeros(x.shape)

    dout_row = dout.reshape(C, outH * outW)
    neuron = 0
    for i in range(0, H - PH + 1, stride):
        for j in range(0, W - PW + 1,stride):
            pool_region = x[:, i:i + PH, j:j + PW].copy().reshape(C, PH * PW)
            dmax_pool = np.zeros(pool_region.shape)

            for idx in range(C):
                max_pool_idx = int(pool_region[idx,:].argmax())
                dout_cur = dout_row[:, neuron]
                dmax_pool[idx, max_pool_idx] = dout_cur[0]
            neuron +=1
            dx[:, i:i + PH, j:j + PW] += dmax_pool.copy().reshape(C, PH, PW)
    return dx


if __name__ == '__main__':
    np.random.seed(42)
    desc = {"type": "maxPooling",
            "pool_size": 2}
    l = MaxPoolLayer(desc)
    inp = (np.random.randn(1, 8, 8) * 10).astype(int)
    import time

    print(inp)
    x = l.forward(inp)
    s_t = time.time()
    zzz = l.derive(x)
    print(time.time() - s_t)
