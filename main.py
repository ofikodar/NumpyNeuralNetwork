import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import Model

data_path = 'data/'
train_data_file = data_path + 'train.csv'
validation_data_file = data_path + 'validate.csv'
test_data_file = data_path + 'test.csv'
hyper_params_path = 'experiments.json'

NUM_CATEGORIES = 10


def load_data():
    train_df = pd.read_csv(train_data_file, header=None, nrows=2000)
    val_df = pd.read_csv(validation_data_file, header=None, nrows=200)
    test_df = pd.read_csv(test_data_file, header=None, nrows=100)

    X_train, y_train = _extract_data(train_df)
    X_val, y_val = _extract_data(val_df)
    X_test, _ = _extract_data(test_df, is_test=True)

    return X_train, y_train, X_val, y_val, X_test, test_df


def _extract_data(df, is_test=False):
    x_data = df[list(df)[1:]].values
    image_size = 32
    image_size_squared = image_size ** 2
    num_channels = 3
    images = np.zeros([len(x_data), num_channels, image_size, image_size])
    for c in range(num_channels):
        images_channel = x_data[:, image_size_squared * c:image_size_squared * (c + 1)]
        images_channel = images_channel.reshape(-1, image_size, image_size)
        images[:, c, :, :] = images_channel
    # images = images.reshape(len(x_data), -1)
    # plt.imshow(images[0])
    # plt.show()
    y_data = None
    if not is_test:
        y_data = np.eye(NUM_CATEGORIES)[df[0] - 1]
    return images, y_data


def _predict_test(model, name):
    predictions = []
    for x in X_test:
        p = model.predict(x).argmax()
        predictions.append(p)

    test_df[0] = predictions
    test_df[0] += 1
    experiment_test_path = test_data_file.replace('.csv', f'_{name}.csv')
    test_df.to_csv(experiment_test_path, index=False, header=None)


def run_experiments():
    experiments_hyper_params = json.load(open(hyper_params_path))
    for params in experiments_hyper_params:
        name = params.pop('name')
        layers_dims = params.pop('layers')
        layers_dims[-2]['num_nodes'] = NUM_CATEGORIES

        model = Model(X_train.shape[1], layers_dims, model_name=name)
        model.fit(X_train, X_val, y_train, y_val, **params)
        _predict_test(model, name)


if __name__ == '__main__':
    np.random.seed(0)
    X_train, y_train, X_val, y_val, X_test, test_df = load_data()
    run_experiments()
