import numpy as np


class BNLayer:
    def __init__(self, layer_description):
        self.type = layer_description['type']
        num_nodes = layer_description['num_nodes']
        prv_layer_shape = layer_description['previous_nodes']

        weights = np.random.randn(num_nodes, prv_layer_shape) / np.sqrt(num_nodes)
        bias = np.zeros(shape=[num_nodes, 1])
        self.weights = weights
        self.bias = bias
        self.running_mean = 0
        self.running_var = 0
        self.eps = 1e-5
        self.momentum = 0.9
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


def _compiled_forward(layer_input, gamma, beta, mode,running_mean,running_var,eps,momentum):
    N, C, H, W = layer_input.shape

    x_flat = layer_input.transpose(0, 2, 3, 1).reshape(-1, C)

    D = C

    out, cache = None, None
    if mode == 'train':
        # Compute output
        mu = x_flat.mean(axis=0)
        xc = x_flat - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + eps)
        xn = xc / std
        out = gamma * xn + beta

        cache = (mode, x_flat, gamma, xc, std, xn, out)

        # Update running average of mean
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        # Update running average of variance
        running_var *= momentum
        running_var += (1 - momentum) * var
    else:
        # Using running mean and variance to normalize
        std = np.sqrt(running_var + eps)
        xn = (x_flat - running_mean) / std
        out = gamma * xn + beta
        cache = (mode, x_flat, xn, gamma, beta, std)

    # Store the updated running means back into bn_param

    running_mean = running_mean
    running_var = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta
