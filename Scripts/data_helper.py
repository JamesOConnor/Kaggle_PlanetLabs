import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import glob


class DataHelper:

    def __init__(self):

        self.train_path = '../input/train_v2.csv'
        self.test_path = '../input/test.csv'
        self.train_dir = '../../train-jpg/' # make sure there's a / at the end of this
        self.test_dir = '../../test-jpg/'

        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        self.weather_labels = ['clear', 'cloudy', 'partly_cloudy', 'haze']
        self.ground_labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'conventional_mine',
                         'cultivation', 'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = np.unique(flatten([l.split(' ') for l in self.train.tags]))
        self.label_map = {l: i for i, l in enumerate(labels)}

    def load_train_images(self, image_size=(32,32)):

        images = []

        for filename, tags in tqdm(self.train.values, miniters=1000):
            filepath = '{}{}.jpg'.format(self.train_dir, filename)
            image = cv2.imread(filepath)
            image = cv2.resize(image, image_size)
            images.append(image)

        images = np.array(images, np.float16) / 255
        print("Loaded training images. Size consumed by arrays {} mb".format((images[0].nbytes) * len(images) / 1024 / 1024))

        label_mat = self.get_label_matrix()

        return images, label_mat

    def load_test_images(self, image_size=(32,32)):

        images = []

        for filename, labels in tqdm(self.test.values, miniters=1000):
            filepath = '{}{}.jpg'.format(self.test_dir, filename)
            image = cv2.imread(filepath)
            image = cv2.resize(image, image_size)
            images.append(image)

        images = np.array(images, np.float16) / 255
        print("Loaded testing images. Size consumed by arrays {} mb".format((images[0].nbytes) * len(images) / 1024 / 1024))

        return images

    def get_label_matrix(self):

        L = []
        for tags in self.train.tags:
            L.append(self.tags_to_onehot(tags))

        D = pd.DataFrame(L, dtype=np.uint8)
        D.columns = sorted(self.label_map)
        return D

    def get_weather_labels(self):

        return self.get_label_matrix[self.weather_labels]


    def predictions_to_submissions(self, predictions, labels, test):
        '''
        Converts outputs from CNNs to file formatted for submission to Kaggle
        :param oh_array: array with binary values representing tags
        :return: array to be written to file
        '''

        out_formatted = []

        for index, image_name in enumerate(test.image_name):
            tags_index = np.where(predictions[index] == 1)[0]

            tags = ' '.join(labels[tags_index])

            out_formatted.append([image_name, tags])

        return pd.DataFrame(out_formatted, columns=['image_name', 'tags'])


    def process_predictions(self, predictions, prediction_threshold):

        """
        Selects one weather label using max prediction value (they're mutually exclusive)
        Thresholds ground labels
        :param predictions:
        :param prediction_threshold:
        :return: np.array of boolean values
        """

        processed_predictions = np.zeros_like(predictions, dtype=np.uint8)

        weather_cols = [self.label_map[x] for x in self.weather_labels]
        ground_cols = [self.label_map[x] for x in self.ground_labels]

        # Awkward but I've spent too long trying to reduce to something neater so fuggedaboutit
        for out, pred in zip(processed_predictions, predictions):

            wc = out[weather_cols]
            wc[np.where(pred[weather_cols] == np.max(pred[weather_cols]))] = 1
            out[weather_cols] = wc

            gc = out[ground_cols]
            gc[np.where(pred[ground_cols] > prediction_threshold)] = 1
            out[ground_cols] = gc

        return processed_predictions

    def calculate_metrics(self, predictions, actual):

        """
        Compares prediction vector with actual vector, returning a number of classification metrics
        :param predictions:
        :param actual:
        :return: metrics
        """

        predictions.dtype = np.bool
        actual.dtype = np.bool

        N = len(predictions) * len(predictions[0])

        TP = np.sum(np.bitwise_and(predictions, actual))
        FP = np.sum(np.bitwise_and(np.invert(predictions), np.invert(actual) ))
        FN = np.sum(np.bitwise_and(predictions, np.invert(actual)))
        TN = np.sum(np.bitwise_and(np.invert(predictions), (actual)))

        correct = np.sum(predictions == actual) / N
        accuracy = (TP + TN) / N
        precision = TP / (TP + FP) # positive predictive value
        sensitivity = TP / (TP + FN) # true positive rate
        specificity = TN / (TN + FP) # true negative rate

        return correct, accuracy, precision, sensitivity, specificity

    def tags_to_onehot(self, tags):

        y = np.zeros(17)
        for i in tags.split(' '):
            y[self.label_map[i]] = 1

        return y

    # def test_batch_generator(self, batch_size=12, image_size=(32,32)):
    #
    #     test = pd.read_csv(self.test_path)
    #
    #     while True:
    #
    #         batch_idx = 0
    #         X = []
    #
    #         for filename, labels in test.values:
    #
    #             filepath = '../../test-jpg/{}.jpg'.format(filename)
    #
    #             img = cv2.imread(filepath)
    #             img = cv2.resize(img, image_size)
    #
    #             X.append(img)
    #             batch_idx += 1
    #
    #             if batch_idx == batch_size:
    #                 yield (np.array(X, np.float16) / 255)
    #                 batch_idx = 0
    #                 X = []

    def load_train_haralick(self):
        data = pd.read_csv('../data/train_colourharalick.csv')
        return data

    def load_test_haralick(self):
        data = pd.read_csv('../data/test_colourharalick.csv')
        return data

    def batch_generator(self, batch_indices, batch_size, shuffle=True, image_size=(32,32), augment_image=False, columns_subset=None):

        if shuffle:
            np.random.shuffle(batch_indices)

        batch_set = self.train.loc[batch_indices, :]
        batch_set_labels = self.get_label_matrix().loc[batch_indices, :]

        # Subset output class labels
        if columns_subset is not None:
            batch_set_labels = batch_set_labels[columns_subset]

        aug = self.load_train_haralick()
        batch_set_aug = aug.loc[batch_indices,:]

        batch_count = 0
        max_count = len(batch_indices)

        total_idx, batch_idx, X, Y, A = 0,0,[],[],[]

        for filename, aug, y in zip(batch_set.image_name, batch_set_aug.values, batch_set_labels.values):

            filepath = '{}{}.jpg'.format(self.train_dir, filename)
            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            if augment_image:
                img = self.augment_image(img, horizontal_flip=False, vertical_flip=False, random_rotate=False)

            X.append(img)
            Y.append(y)
            A.append(aug)
            batch_idx += 1
            total_idx += 1

            if batch_idx == batch_size or total_idx == max_count:
                yield (np.array(X, np.float16) / 255, np.array(A), np.array(Y, np.uint8))
                batch_idx, X, Y, A = 0, [], [], []
                batch_count += 1


    def test_batch_generator(self, batch_indices, batch_size, shuffle=False, image_size=(32,32), augment_image=False, columns_subset=None):

        if shuffle:
            np.random.shuffle(batch_indices)

        batch_set = self.test.loc[batch_indices, :]
        batch_set_labels = self.get_label_matrix().loc[batch_indices, :]

        if columns_subset is not None:
            batch_set_labels = batch_set_labels[columns_subset]

        aug = self.load_test_haralick()
        batch_set_aug = aug.loc[batch_indices,:]

        batch_count = 0
        max_count = len(batch_indices)

        total_idx, batch_idx, X, Y, A = 0,0,[],[],[]

        for filename, aug in zip(batch_set.image_name, batch_set_aug.values, batch_set_labels.values):

            filepath = '{}{}.jpg'.format(self.train_dir, filename)
            img = cv2.imread(filepath)
            img = cv2.resize(img, image_size)

            if augment_image:
                img = self.augment_image(img, horizontal_flip=False, vertical_flip=False, random_rotate=False)

            X.append(img)
            A.append(aug)
            batch_idx += 1
            total_idx += 1

            if batch_idx == batch_size or total_idx == max_count:
                yield (np.array(X, np.float16) / 255, np.array(A))
                batch_idx, X, A = 0, [], []
                batch_count += 1

    def train_batch_generator(self, batch_size=12, image_size=(32,32), horizontal_flip=False, vertical_flip=False, random_rotate=False):

        batch_count = 0

        while True:

            batch_idx = 0
            X = []
            Y = []

            for filename, tags in self.train.values:

                filepath = '../../train-jpg/{}.jpg'.format(filename)

                img = cv2.imread(filepath)
                img = cv2.resize(img, image_size)

                img = self.augment_image(img, horizontal_flip=False, vertical_flip=False, random_rotate=False)

                y = np.zeros(17)
                for i in tags.split(' '):
                    y[self.label_map[i]] = 1

                X.append(img)
                Y.append(y)
                batch_idx += 1

                if batch_idx == batch_size:
                    yield(np.array(X, np.float16) / 255, np.array(Y, np.uint8))
                    batch_idx = 0
                    batch_count += 1
                    X = []
                    Y = []


    """
                    Image augmentation functions 
    """

    def augment_image(self, image, horizontal_flip=True, vertical_flip=True, random_rotate=True):

        rot_map = {0: 0, 1:90, 2:180, 3:270}

        # Coin toss on horizontally flipping the image
        if horizontal_flip:
            if random.randint(0, 1):
                image = self.hor_flip_image(image)

        # Coin toss on vertically flipping the image
        if vertical_flip:
            if random.randint(0, 1):
                image = self.ver_flip_image(image)

        # Randomly rotate the image
        if random_rotate:
            rot_angle = rot_map[random.randint(0, 3)]
            image = self.rotate_image(image, rot_angle)

        return image

    def rotate_image(self, image, angle):

      rows, cols, channels = image.shape
      image_center = (int(rows / 2), int(cols / 2))

      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, (rows, cols))

      return result

    def hor_flip_image(self, image):
        result = cv2.flip(image, 1)
        return result

    def ver_flip_image(self, image):
        result = cv2.flip(image, 0)
        return result