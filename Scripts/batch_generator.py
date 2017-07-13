import pandas as pd
import numpy as np
import cv2
import tqdm
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import glob

# Creates a batch generator for training images ..

def flow_generator(directory):

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode=None)


def test_batch_generator(batch_size=12, image_size=(32,32)):

    filepaths = glob.glob('../../test-jpg/*.jpg')

    while True:
        batch_idx = 0
        X = []

        for filepath in filepaths:

            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            X.append(img)
            batch_idx += 1

            if batch_idx == batch_size:
                yield (np.array(X, np.float16) / 255)
                batch_idx = 0
                X = []

def train_batch_generator(batch_size=12, image_size=(32,32)):

    train = pd.read_csv('train.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

    label_map = {l: i for i, l in enumerate(labels)}

    while True:

        batch_idx = 0
        X = []
        Y = []

        for filename, labels in train.values:

            filepath = '../../train-jpg/{}.jpg'.format(filename)

            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            y = np.zeros(17)
            for i in labels.split(' '):
                y[label_map[i]] = 1

            X.append(img)
            Y.append(y)
            batch_idx += 1

            if batch_idx == batch_size:
                yield(np.array(X, np.float16) / 255, np.array(Y, np.uint8))
                batch_idx = 0
                X = []
                Y = []
