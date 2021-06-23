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
