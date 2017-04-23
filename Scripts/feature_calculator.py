import numpy as np
import scipy as sp, scipy.stats as sps
import mahotas as mh


def calc_hist_stats(X):
    """
    Calculates histogram statistics on 1-D input array X.  
    :param X:  
    :return: list [min, mean, median, max, std, energy, entropy, skewness, kurtosis]
    """

    xmin = sp.amin(X)
    mean = sp.mean(X)
    median = sp.median(X)
    xmax = sp.amax(X)
    std = sp.std(X)
    energy = np.sum(X**2)
    entropy = sps.entropy(X)
    skewness = sps.skew(X)
    kurtosis = sps.kurtosis(X)

    return xmin, mean, median, xmax, std, energy, entropy, skewness, kurtosis


def calc_img_hist_stats(im):
    """
    Calculates histogram statistics for RGB  for all bands using numpy
    :param im: 
    :return: list of RGB histogram statistics 
    """

    r_hist = calc_hist_stats(im[:,:,0].flatten())
    g_hist = calc_hist_stats(im[:,:,1].flatten())
    b_hist = calc_hist_stats(im[:,:,2].flatten())

    return r_hist + g_hist + b_hist


def calc_img_haralick(im):
    """
    Calculates haralick features for RGB input image
    :param im: 
    :return: list of RGB haralick features 
    """

    # Summed and averaged because we're not hooligans
    R_feats = sum(mh.features.haralick(im[:,:,0]))/4
    G_feats = sum(mh.features.haralick(im[:,:,1]))/4
    B_feats = sum(mh.features.haralick(im[:,:,2]))/4

    return R_feats + G_feats + B_feats


def get_haralick_labels():

    return ['R_haralick1', 'R_haralick2', 'R_haralick3', 'R_haralick4', 'R_haralick5', 'R_haralick6', 'R_haralick7',
            'R_haralick8', 'R_haralick9', 'R_haralick10', 'R_haralick11', 'R_haralick12', 'R_haralick13',
            'G_haralick1', 'G_haralick2', 'G_haralick3', 'G_haralick4', 'G_haralick5', 'G_haralick6', 'G_haralick7',
            'G_haralick8', 'G_haralick9', 'G_haralick10', 'G_haralick11', 'G_haralick12', 'G_haralick13',
            'B_haralick1', 'B_haralick2', 'B_haralick3', 'B_haralick4', 'B_haralick5', 'B_haralick6', 'B_haralick7',
            'B_haralick8', 'B_haralick9', 'B_haralick10', 'B_haralick11', 'B_haralick12', 'B_haralick13']


def get_colour_stats_labels():

    return ['R_min', 'R_mean', 'R_median', 'R_max', 'R_std', 'R_energy', 'R_entropy', 'R_skewness', 'R_kurtosis',
            'G_min', 'G_mean', 'G_median', 'G_max', 'G_std', 'G_energy', 'G_entropy', 'G_skewness', 'G_kurtosis',
            'B_min', 'B_mean', 'B_median', 'B_max', 'B_std', 'B_energy', 'B_entropy', 'B_skewness', 'B_kurtosis']
