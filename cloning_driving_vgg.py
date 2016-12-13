"""
Build a model that clones human driving behavior using a simple VGG like
neural net architecture
"""

import argparse
import json

import matplotlib.image as mpimg
import numpy as np
import scipy.misc

import tensorflow as tf

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D,\
    Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

import config_vgg as config
import utils


LOG_FILE_PATH = './driving_log.csv'
CENTER_CAM_FILE_IDX = 0
STEERING_ANGLE_INDX = 3


def build_model():
    model = Sequential()

    model.add(Convolution2D(config.conv1['nb_filter'],
                            config.conv1['nb_row'],
                            config.conv1['nb_col'],
                            border_mode='same',
                            input_shape=(config.img['height'],
                                         config.img['width'],
                                         config.img['num_channels']),
                            W_regularizer=l2(config.l2_regularization_scale),
                            b_regularizer=l2(config.l2_regularization_scale)))
    model.add(MaxPooling2D(pool_size=(config.pool1['nb_row'],
                                      config.pool1['nb_col']),
                           border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Convolution2D(config.conv2['nb_filter'],
                            config.conv2['nb_row'],
                            config.conv2['nb_col'],
                            border_mode='same',
                            W_regularizer=l2(config.l2_regularization_scale),
                            b_regularizer=l2(config.l2_regularization_scale)))
    model.add(MaxPooling2D(pool_size=(config.pool2['nb_row'],
                                      config.pool2['nb_col']),
                           border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Convolution2D(config.conv3['nb_filter'],
                            config.conv3['nb_row'],
                            config.conv3['nb_col'],
                            border_mode='same',
                            W_regularizer=l2(config.l2_regularization_scale),
                            b_regularizer=l2(config.l2_regularization_scale)))
    model.add(MaxPooling2D(pool_size=(config.pool3['nb_row'],
                                      config.pool3['nb_col']),
                           border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Flatten())

    model.add(Dense(config.fc1['nb_units'],
                    W_regularizer=l2(config.l2_regularization_scale),
                    b_regularizer=l2(config.l2_regularization_scale)))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Dense(config.fc2['nb_units'],
                    W_regularizer=l2(config.l2_regularization_scale),
                    b_regularizer=l2(config.l2_regularization_scale)))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Dense(config.fc3['nb_units'],
                    W_regularizer=l2(config.l2_regularization_scale),
                    b_regularizer=l2(config.l2_regularization_scale)))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Dense(config.fc4['nb_units'],
                    W_regularizer=l2(config.l2_regularization_scale),
                    b_regularizer=l2(config.l2_regularization_scale)))
    model.add(Activation('relu'))
    model.add(Dropout(config.keep_prob))

    model.add(Dense(config.fc5['nb_units'],
                    W_regularizer=l2(config.l2_regularization_scale),
                    b_regularizer=l2(config.l2_regularization_scale)))

    model.compile(loss='mse',
                  optimizer=Adam(lr=config.learning_rate))

    return model


def get_data_stats(fs_path):
    def _func(record):
        d = record.strip().split(',')
        return float(d[STEERING_ANGLE_INDX])
    with open(fs_path, 'r') as f:
        data = f.read().strip().split('\n')
        data = list(map(_func, data))
        nb_samples = len(data)
        going_left = np.asarray(list(filter(lambda x: x < 0., data)))
        going_right = np.asarray(list(filter(lambda x: x > 0., data)))
    return {
        'nb_samples': nb_samples,
        'left': {
            'nb_samples': len(going_left),
            'min': going_left.min(),
            'median': np.median(going_left),
            'mean': np.mean(going_left),
        },
        'right': {
            'nb_samples': len(going_right),
            'max': going_right.max(),
            'median': np.median(going_right),
            'mean': np.mean(going_right),
        },
        'center': {
            'nb_samples': len(data) - len(going_left) - len(going_right)
        },
    }


def get_driving_data(fs_path):
    f = open(fs_path, 'r')
    lines = f.read().strip().split('\n')
    f.close()
    X, y = [], []
    for line in lines:
        parts = line.strip().split(',')
        img = mpimg.imread(parts[CENTER_CAM_FILE_IDX])
        X.append(utils.normalize(scipy.misc.imresize(
            img, (config.img['height'], config.img['width'])
        )))
        y.append(float(parts[STEERING_ANGLE_INDX]))
    return {
        'X': np.asarray(X),
        'y': np.asarray(y)
    }


def train_model(model, training_data, nb_epoch):
    return model.fit(training_data['X'], training_data['y'],
                     batch_size=config.training_batch_size,
                     nb_epoch=nb_epoch)


def test_model(model, test_data):
    return model.evaluate(test_data['X'], test_data['y'],
                          batch_size=config.test_batch_size)


def predict(model, X):
    return model.predict(X, batch_size=config.prediction_batch_size)


def save_model(model):
    with open('model.json', 'w') as model_json_file:
        json.dump(model.to_json(), model_json_file)
    model.save_weights('model.h5')


def load_model():
    with open('model.json', 'r') as model_json_file:
        model = model_from_json(json.load(model_json_file))
        model.load_weights('model.h5')
    return model


def main(log_file_path, save_to_disk=True, load_from_disk=False):
    if load_from_disk:
        model = load_model()
    else:
        model = build_model()
    training_data = get_driving_data(log_file_path)
    train_model(model, training_data, nb_epoch=config.nb_epoch)
    if save_to_disk:
        save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training behavioral cloning'
                                                 'model for autonomous driving')
    parser.add_argument('--log', required=False,
                        default=LOG_FILE_PATH,
                        help='path to the file with training data log')
    parser.add_argument('--save', action='store_true', default=True,
                        help='flag to control saving the model to disk')
    parser.add_argument('--load', action='store_true',
                        help='flag to control loading a pre-trained model')
    args = parser.parse_args()
    main(args.log, args.save, args.load)