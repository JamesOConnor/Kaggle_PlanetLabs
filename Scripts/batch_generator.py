import pandas as pd
import numpy as np
import cv2
import tqdm
from keras.datasets import mnist
from keras.utils import np_utils

# Creates a batch generator for training images ..

def batch_generator(batch_size=128):

    # Read label file and generate label_map
    train = pd.read_csv('train.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

    label_map = {l: i for i, l in enumerate(labels)}

    while True:

        batch_idx = 0
        X = []
        Y = []

        for idx, row in enumerate(train.values):

            img = cv2.imread('../../train-jpg/{}.jpg'.format(idx))
            img = cv2.resize(img, 128, 128)
            tags = row[1]
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1

            X.append(img)
            Y.append(targets)
            batch_idx += 1

            if batch_idx == batch_size:
                yield(np.array(X, np.float16) / 255, np.array(Y, np.uint8) )
                batch_idx = 0
                X = []
                Y = []
