import cv2
import numpy as np
import mahotas as mh


def calc_haralick():
    '''
  Calculates and returns haralick features for all bands using mahotas library
  '''
    train = np.loadtxt('train.csv', delimiter=',', dtype=str)
    mh_feats_green = np.zeros((len(train[1:, 0]), 52))
    mh_feats_red = np.zeros((len(train[1:, 0]), 52))
    mh_feats_blue = np.zeros((len(train[1:, 0]), 52))

    for i in range(len(train[1:])):
        if i % 50 == 0:
            print i
        im = cv2.imread('train-jpg/train_%s.jpg' % (int(i)))
        mh_feats_green[i] = np.hstack(mh.features.haralick(im[:, :, 1]))
        mh_feats_red[i] = np.hstack(mh.features.haralick(im[:, :, 0]))
        mh_feats_blue[i] = np.hstack(mh.features.haralick(im[:, :, 2]))
    return np.r_['1,2,0', mh_feats_green, mh_feats_red, _mh_feats_blue]
