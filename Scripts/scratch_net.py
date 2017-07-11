import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm
import glob
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from oh_to_labs import *
from batch_generator import *


def get_model():

    model = Sequential()

    model.add(Conv2D(128,
                     input_shape=(128, 128, 3),
                     kernel_size=(5, 5),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64,
                     kernel_size=(5, 5),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(17, activation='sigmoid'))

    return model

if __name__ == '__main__':

    train = pd.read_csv('train.csv')

    # flatten = lambda l: [item for sublist in l for item in sublist]
    # labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
    #
    # label_map = {l: i for i, l in enumerate(labels)}
    # inv_label_map = {i: l for l, i in label_map.items()}
    #
    # x_train = []
    # y_train = []
    #
    # print('Reading image files')
    # for f, tags in tqdm(train.values, miniters=1000):
    #
    #     img = cv2.imread('../../train-jpg/{}.jpg'.format(f))
    #     targets = np.zeros(17)
    #     for t in tags.split(' '):
    #         targets[label_map[t]] = 1
    #     x_train.append(cv2.resize(img, (128, 128)))
    #     y_train.append(targets)
    #
    # y_train = np.array(y_train, np.uint8)  # ints for NN training set
    # x_train = np.array(x_train, np.float16) / 255.  # Convert to floats for CNN input

    # Model definition
    with tf.device('/gpu:0'):
        model = get_model()

    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])

    start = time.time()

    batch_size = 24
    batch_gen = image_batch_generator(batch_size)

    steps_per_epoch = int(len(train) / batch_size) + 1

    model.fit_generator(batch_gen, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1)

    end = time.time()

    model_path = 'test.model'
    model.save(model_path)

    print('Training time: %s' % str(end-start))

    # fns = glob.glob('../../test-jpg/*.jpg')
    # x_test = []
    #
    # # for fn in tqdm(fns, miniters=1000):
    # #     img = cv2.imread(fn)
    # #     targets = np.zeros(17)
    # #     for t in tags.split(' '):
    # #         targets[label_map[t]] = 1
    # #     x_test.append(cv2.resize(img, (64, 64)))
    #
    # p_valid = model.predict(x_test, batch_size=128)
    #
    # p_valid_classes = np.zeros_like(p_valid)
    # p_valid_classes[np.where(p_valid>.2)] = 1
    # for_sub = oh_to_labs(p_valid_classes)

