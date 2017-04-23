import multiprocessing as mp
import cv2 as cv
import pandas as pd
import calc_colour_stats as cc
import numpy as np


def process_images(x):
    filename = '../../train-jpg/train_%s.jpg' % (int(x))
    im = cv.imread(filename)
    y = cc.calc_img_hist_stats(im)
    return [x] + list(y)

if __name__ == '__main__':

    print('Reading train.csv')
    train = pd.read_csv('../../train.csv')
    N = len(train.image_name)
    labels = cc.get_column_labels()

    print('Processing images')
    pool = mp.Pool(processes=4)
    data = np.ndarray(shape=(N, len(labels)+1))
    for index, result in enumerate(pool.imap(process_images, range(N), chunksize=20)):
        data[index,:] = result

    pool.close()
    pool.join()

    print('Writing dataframe')
    df = pd.DataFrame(data=data[0:, 1:], index=data[0:, 0], columns=labels)

    df.to_csv('../data/train_colourhist.csv')

    print('Completed')
