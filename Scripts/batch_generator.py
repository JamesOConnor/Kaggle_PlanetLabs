import pandas as pd
import numpy as np
import cv2
import random
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

def rotate_image(image, angle):

  rows, cols, channels = image.shape
  image_center = (int(rows / 2), int(cols / 2))

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, (rows, cols))

  return result

def hor_flip_image(image):
    result = cv2.flip(image, 1)
    return result

def ver_flip_image(image):
    result = cv2.flip(image, 0)
    return result

def test_batch_generator(batch_size=12, image_size=(32,32)):

    test = pd.read_csv('test.csv')

    while True:

        batch_idx = 0
        X = []

        for filename, labels in test.values:

            filepath = '../../test-jpg/{}.jpg'.format(filename)

            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            X.append(img)
            batch_idx += 1

            if batch_idx == batch_size:
                yield (np.array(X, np.float16) / 255)
                batch_idx = 0
                X = []

def train_batch_generator(batch_size=12, image_size=(32,32), horizontal_flip=False, vertical_flip=False, random_rotate=False):

    train = pd.read_csv('train.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = np.unique(flatten([l.split(' ') for l in train.tags]))
    label_map = {l: i for i, l in enumerate(labels)}
    rot_map = {0: 0, 1:90, 2:180, 3:270}

    batch_count = 0
    while True:

        batch_idx = 0
        X = []
        Y = []

        for filename, labels in train.values:

            filepath = '../../train-jpg/{}.jpg'.format(filename)

            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            # Coin toss on horizontally flipping the image
            if horizontal_flip:
                if random.randint(0,1):
                    img = hor_flip_image(img)

            # Coin toss on vertically flipping the image
            if vertical_flip:
                if random.randint(0,1):
                    img = ver_flip_image(img)

            # Randomly rotate the image
            if random_rotate:
                rot_angle = rot_map[random.randint(0,3)]
                img = rotate_image(img, rot_angle)

            y = np.zeros(17)
            for i in labels.split(' '):
                y[label_map[i]] = 1

            X.append(img)
            Y.append(y)
            batch_idx += 1

            if batch_idx == batch_size:
                yield(np.array(X, np.float16) / 255, np.array(Y, np.uint8))
                batch_idx = 0
                batch_count += 1
                X = []
                Y = []
