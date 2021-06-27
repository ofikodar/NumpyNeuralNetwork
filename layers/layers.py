from layers.activations import Sigmoid, LeakyRelu, Softmax
from layers.conv2d import Conv2DLayer
from layers.fully_connected import FCLayer
from layers.reshape import FlattenLayer, MaxPoolLayer

layers_type_dict = dict()
layers_type_dict['fc'] = FCLayer
layers_type_dict['conv2d'] = Conv2DLayer
layers_type_dict['flatten'] = FlattenLayer
layers_type_dict['sigmoid'] = Sigmoid
layers_type_dict['leakyRelu'] = LeakyRelu
layers_type_dict['softmax'] = Softmax
layers_type_dict['maxPooling'] = MaxPoolLayer
