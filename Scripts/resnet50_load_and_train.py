import numpy as np
import keras
import cv2
import glob
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
import pandas as pd

'''
A script for training and generating results of a CNN using ResNet 50's architecture
'''

def oh_to_labs(oh_array):
    '''
    Converts outputs from CNNs to file formatted for submission to Kaggle
    :param oh_array: array with binary values representing tags
    :return: array to be written to file
    '''

    train = pd.read_csv('train.csv')

    fns = glob.glob('test-jpg/*.jpg')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
    labels.sort()
    labels = np.array(labels)
    out_formatted = []
    out_formatted.append(['image_name', 'tags'])
    for n,i in enumerate(fns):
        tags = np.where(oh_array[n]==1)[0]
        final_tags = ' '.join(labels[tags])
        out_formatted.append([i.split('\\')[1].split('.')[0], final_tags])
    return np.array(out_formatted)


def make_and_train_model(imsize, save=True):
    '''
    Trains and returns a CNN with the ResNet50 architecture
    :param imsize: Integer, size of input images, will resize to this value on both axes
    :return: Trained CNN
    '''
    x_train = []
    y_train = []
    im_size = imsize

    train = pd.read_csv('train.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
    labels.sort()
    labels = np.array(labels)

    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    for f, tags in tqdm(train.values, miniters=1000):
        img = cv2.imread('train-jpg/{}.jpg'.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, (im_size, im_size)))
        y_train.append(targets)

    y_train = np.array(y_train, np.uint8)  # ints for NN training set
    x_train = np.array(x_train, np.float16) / 255.  # Convert to floats for CNN input

    xt = x_train
    xt2 = np.rot90(xt, axes=(1,2))
    xt3 = np.rot90(xt2, axes=(1,2))
    xt4 = np.rot90(xt3, axes=(1,2))
    x_train = np.r_['0', xt, xt2, xt3, xt4]
    y_train = np.r_['0', y_train, y_train, y_train, y_train]
    #split = 128000
    #x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

    model = ResNet50(classes=17)
    results = []
    # ~13 epochs works decent

    tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True,
                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    if 'x_valid' in locals():
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=1,
                  verbose=1,
                  validation_data=(x_valid, y_valid), callbacks=[tb])
    else:
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=1,
                  verbose=1,
                  callbacks=[tb])
    if save == True:
        model.save('ResNet50_CNN.h5')
    return model


def fit_and_prep_results(model, imsize, thresh=.2):
    '''
    Takes an input model with a specified image size and returns results against training set ready for submission
    :param model: CNN
    :param imsize: Integer, size of input images, will resize to this value on both axes
    :param thresh: Float, threshold for tags to be attached to an image
    :return:
    '''
    fns = glob.glob('test-jpg/*.jpg')
    x_test = []

    for fn in tqdm(fns, miniters=1000):
        img = cv2.imread(fn)
        x_test.append(cv2.resize(img, (imsize, imsize)))

    x_test = np.array(x_test, np.float16) / 255.
    p_valid = model.predict(x_test)
    p_valid_classes = np.zeros_like(p_valid)
    p_valid_classes[np.where(p_valid > thresh)] = 1
    for_sub = oh_to_labs(p_valid_classes)
    return for_sub


if __name__ == '__main__':
    model = make_and_train_model(256)
    results = fit_and_prep_results(model, 256)
    np.savetxt('for_submission.csv', results, delimiter=',')