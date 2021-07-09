import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

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

    def predict(self, x):
        a = x
        for i in range(self.num_layers):
            a = self.layers[str(i)].forward(a)
        prediction = a
        return prediction

    def backprop(self, x, y):
        dz = self.layers[str(self.last_layer_index)].derive(y)
        for i in range(1, self.num_layers):
            dz = self.layers[str(self.last_layer_index - i)].derive(dz)

    def update_weights(self, lr):
        for layer in range(self.num_layers):
            self.layers[str(layer)].update_weights(lr)
            #
            # try:
            #     print(self.layers[str(layer)].type)
            #     print(np.array(self.layers[str(layer)].weights_gradients).shape)
            #     print(np.array(self.layers[str(layer)].weights).shape)
            #     self.layers[str(layer)].update_weights(lr)
            #
            # except:
            #     pass
            # print("------------------")
    def cross_entropy(self, predictions, labels):

        cost = _compiled_cross_entropy(predictions, labels)
        return cost

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

            epoch_correct_samples = 0
            epoch_count_samples = 0

            epoch_aggregated_log = 0

            np.random.shuffle(idx)
            _X_train = X_train[idx]
            _y_train = y_train[idx]

            for batch_idx in range(num_batches):
                batch_x = _X_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_y = _y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                for x, y in zip(batch_x, batch_y):
                    y = np.expand_dims(y, axis=1)
                    if len(x.shape) == 1:
                        x = np.expand_dims(x, axis=1)
                    p = self.predict(x)
                    self.backprop(x, y)
                    epoch_aggregated_log, epoch_correct_samples = self.update_epoch_training_history(p, y,
                                                                                                     epoch_aggregated_log,
                                                                                                     epoch_correct_samples)
                    epoch_count_samples+=1
                self.update_weights(lr)
            train_loss, train_acc = self.train_report(epoch_count_samples, epoch_aggregated_log, epoch_correct_samples)
            # val_loss, val_acc = self.val_report(X_train, y_train)

            val_loss, val_acc = self.val_report(X_test, y_test)

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

    def update_epoch_training_history(self, p, y, loss, acc):
        loss += self.cross_entropy(p, np.expand_dims(y, 1))
        if np.argmax(p) == np.argmax(y):
            acc += 1
        return loss, acc

    def train_report(self, epoch_count_samples, loss, acc):
        name = 'Train'
        loss /= epoch_count_samples
        acc /= epoch_count_samples
        acc = 100 * acc

        print(f"{name} loss:", round(loss, 4))
        print(f"{name} Acc:", round(acc, 4))

        return loss, acc

    def val_report(self, x_data, y_data):
        name = "Val"
        loss = 0
        acc = 0
        for x, y in zip(x_data, y_data):
            if len(x.shape) == 1:
                x = np.expand_dims(x, 1)
            p = self.predict(x)
            loss += self.cross_entropy(p, np.expand_dims(y, 1))
            if np.argmax(p) == np.argmax(y):
                acc += 1

        loss /= x_data.shape[0]
        acc /= x_data.shape[0]
        acc = 100 * acc

        print(f"{name} loss:", round(loss, 4))
        print(f"{name} Acc:", round(acc, 4))

        return loss, acc


@jit(nopython=True)
def _compiled_cross_entropy(predictions, labels):
    epsilon = 1e-10
    cost = -np.sum(labels.flatten() * np.log(predictions.flatten() + epsilon))
    return cost
