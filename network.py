import copy
import os

import matplotlib.pyplot as plt
import numpy as np
# from numba import jit

from layers.layers import layers_type_dict


class Model:

    def __init__(self, input_shape, nodes_list, model_name):
        self.model_name = model_name
        self.layers = dict()
        nodes_list[0]['previous_nodes'] = input_shape
        for i, layer_description in enumerate(nodes_list):
            layer_type = layer_description['type']
            layer = layers_type_dict[layer_type](layer_description)
            self.layers[str(i)] = layer

        self.num_layers = len(nodes_list)
        self.last_layer_index = self.num_layers - 1
        self.best_model = self.layers.copy()

    def predict(self, x,is_train=False):
        a = x
        for i in range(self.num_layers):
            layer = self.layers[str(i)]
            a = layer.forward(a,is_train)
        prediction = a
        return prediction

    def backprop(self, y):
        dx = self.layers[str(self.last_layer_index)].derive(y)
        for i in range(1, self.num_layers):
            dx = self.layers[str(self.last_layer_index - i)].derive(dx)

    def update_weights(self, lr):
        for layer in range(self.num_layers):
            self.layers[str(layer)].update_weights(lr)

    def calc_loss(self, predictions, labels):
        batch_size = predictions.shape[0]
        log_prob = np.log(predictions)
        loss = -np.sum(log_prob[np.arange(batch_size), labels]) / batch_size
        return loss

    def fit(self, X_train, X_test, y_train, y_test, batch_size=32, lr=0.01,
            lr_decay=0.8, lr_patience=3, es_patience=7,
            epochs=10):

        train_samples = X_train.shape[0]
        num_batches = train_samples // batch_size

        history = dict()
        history['train_acc'] = []
        history['train_loss'] = []
        history['val_acc'] = []
        history['val_loss'] = []

        idx = list(range(train_samples))

        es_patience_count = 0
        lr_patience_count = 0

        prev_loss = np.inf
        prev_acc = 0

        for e in range(epochs):
            print("Epoch num:", e)

            np.random.shuffle(idx)
            _X_train = X_train[idx]
            _y_train = y_train[idx]

            for batch_idx in range(num_batches):
                batch_x = _X_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_y = _y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                self.predict(batch_x)
                self.backprop(batch_y)
                self.update_weights(lr)

            train_acc, train_loss = self.report(X_test, y_test, name='Train')
            val_loss, val_acc = self.report(X_test, y_test, name='Val')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            print("===================")
            if val_loss > prev_loss:
                es_patience_count += 1
                lr_patience_count += 1
            else:
                prev_loss = val_loss
                prev_acc = val_acc
                es_patience_count = 0
                self.best_model = copy.deepcopy(self.layers)

            if lr_patience_count == lr_patience:
                lr *= lr_decay
                lr_patience_count = 0
                print("LR Decay:", round(lr, 5))
            if es_patience_count == es_patience:
                break
        self.layers = copy.deepcopy(self.best_model)
        print("Finished Training, Best Acc:", round(prev_acc, 4))
        self._export_history(history)

    def _export_history(self, history):
        output_dir = 'experiments/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        keys = ['acc', 'loss']
        total_epochs = len(history['train_loss'])
        idx = range(total_epochs)
        for key in keys:
            plt.title(key.capitalize() + f" - {self.model_name}")
            plt.grid()
            plt.plot(idx, history[f'train_{key}'])
            plt.plot(idx, history[f'val_{key}'])
            plt.legend(['train', 'val'])

            plt.xlabel("Epochs")
            plt.savefig(output_dir + f"{key} - {self.model_name}.jpg")
            plt.clf()

    def report(self, x_data, y_data, name):
        pred = self.predict(x_data)
        loss = self.calc_loss(pred, y_data)
        acc = np.sum(pred.argmax(axis=1) == y_data)
        acc /= x_data.shape[0]
        acc *= 100

        print(f"{name} loss:", round(loss, 4))
        print(f"{name} Acc:", round(acc, 4))

        return loss, acc
