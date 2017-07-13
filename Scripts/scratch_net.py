import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm
import glob
import time

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from oh_to_labs import *

from batch_generator import *
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, EarlyStopping




class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def get_model():

    img_size = (32, 32)
    img_channels = 3
    output_size = 17

    model = Sequential()

    model.add(Conv2D(128,
                     input_shape=(*img_size, img_channels),
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

    model.add(Dense(output_size, activation='sigmoid'))

    return model

def get_amz_model(image_size=(32,32), image_channels=3):

    output_size = 17 # I guess this doesn't change that much

    model = Sequential()

    model.add(BatchNormalization(input_shape=(*image_size, image_channels)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    # def add_flatten_layer(self):
    model.add(Flatten())

    # def add_ann_layer(self, output_size):
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='sigmoid'))

    return model


def oh_to_labs(oh_array, train, test):
    '''
    Converts outputs from CNNs to file formatted for submission to Kaggle
    :param oh_array: array with binary values representing tags
    :return: array to be written to file
    '''

    # TODO: Niceify
    # Todo: Why does the output get returned in a weird order? 0,1,10,100,1000, etc ..
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
    labels = np.array(labels)

    out_formatted = []
    out_formatted.append(['image_name', 'tags'])
    for n,i in enumerate(test):
        tags = np.where(oh_array[n]==1)[0]
        final_tags = ' '.join(labels[tags])
        out_formatted.append([i.split('\\')[1].split('.')[0], final_tags])

    return np.array(out_formatted)


if __name__ == '__main__':

    # Get train and test filepaths
    train = pd.read_csv('train.csv')
    test = glob.glob('../../test-jpg/*.jpg')

    # Some training options
    num_epochs = 5
    batch_size = 12
    verbosity_level = 1
    image_size = (128,128) # inits tensor shapes as well as opencv resize in batch generators
    image_channels = 3
    prediction_threshold = 0.2 # for calling a spade a spade
    fun_loss = 'binary_crossentropy'    # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    cool_optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)     # Adam optimizer with nesterov momentum

    # Model architecture definition
    with tf.device('/gpu:0'):
        model = get_amz_model(image_size=image_size, image_channels=image_channels)

    # Loss function and GD algorithm
    model.compile(loss=fun_loss,
                  optimizer=cool_optimizer,
                  metrics=['accuracy'])

    # This is actually pretty cool
    print(model.summary())

    # TODO: early stopping will auto-stop training process if model stops learning after 3 epochs
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    # For logging
    history = LossHistory()

    # X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size) # TODO

    # TODO: We can put this higher if we introduce data augmentation.
    steps_per_epoch_train = int(len(train) / batch_size) + 1

    # TODO: Use the history callback somehow
    # TODO: Poke around the early stopping parameters
    # TODO: Tensorboard

    # Train model on batch data
    model.fit_generator(train_batch_generator(batch_size, image_size),
                        steps_per_epoch_train,
                        epochs=num_epochs,
                        verbose=1,
                        # callbacks = [history, earlyStopping]
                        # validation_data=(X_valid, y_valid) # TODO using generator somehow .. (pre split the indices and save in class)
                        )

    # Save model
    model_path = 'test_model.h5'
    model.save(model_path)

    # Alternatively ..
    # model = load_model(model_path)

    # Test model

    steps_per_epoch_test = int(len(test) / batch_size) + 1

    predictions = model.predict_generator(test_batch_generator(batch_size, image_size),
                                          steps_per_epoch_test,
                                          verbose=1)

    # p_valid = model.predict(X_valid)
    # fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
    # info = (history.train_losses, history.val_losses, fbeta_score)



    prediction_classes = np.zeros_like(predictions)
    prediction_classes[np.where(predictions>prediction_threshold)] = 1

    outputs = oh_to_labs(prediction_classes, train, test)

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv('submission.csv', header=False, index=False)


