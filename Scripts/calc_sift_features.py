import cv2 as cv
import numpy as np
import pandas as pd


def calc_img_sift_features(im):
    """
    Calculates and returns number of ORB, SURF and SIFT features in an image
    """

    orb_det = cv.ORB()
    sift_det = cv.SIFT()
    surf_det = cv.SURF()

    for i in range(len(train[1:])):
        if i % 50 == 0:
            print(i)

        im = cv2.imread('train-jpg/train_%s.jpg' % (int(i)))
        orb_feats_green[i] = len(orb_det.detect(im[:, :, 1]))
        orb_feats_red[i] = len(orb_det.detect(im[:, :, 0]))
        orb_feats_blue[i] = len(orb_det.detect(im[:, :, 2]))
        sift_feats_green[i] = len(sift_det.detect(im[:, :, 1]))
        sift_feats_red[i] = len(sift_det.detect(im[:, :, 0]))
        sift_feats_blue[i] = len(sift_det.detect(im[:, :, 2]))
        surf_feats_green[i] = len(surf_det.detect(im[:, :, 1]))
        surf_feats_red[i] = len(surf_det.detect(im[:, :, 0]))
        surf_feats_blue[i] = len(surf_det.detect(im[:, :, 2]))

    return np.r_['1,2,0', orb_feats_green, orb_feats_red, orb_feats_blue, sift_feats_green, sift_feats_red, sift_feats_blue, surf_feats_green, sift_feats_red, sift_feats_blue]
