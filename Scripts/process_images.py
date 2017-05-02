import multiprocessing as mp
from functools import partial
import time
import glob
import os
import cv2 as cv
import pandas as pd
import numpy as np
import feature_calculator as fc

"""
Notes:
- process_images prepends x to calculated feature vector .. This was originally due to async processing queue but 
   as the numpy array is now preallocated and results inserted via index, is no longer required (but I don't want to fix) 
"""

def process_images(filename, process_function):

    #
    image_id = os.path.split(filename)[-1]
    image_id = int(image_id.split('.')[0].split('_')[-1]) # sorry

    im = cv.imread(filename)

    if im.shape == (256, 256, 3):
        im = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        y = process_function(im)

    return [image_id] + list(y)


if __name__ == '__main__':

    # Basic run configuration
    image_dir = '../../test-jpg/'
    output_filepath = '../data/test_HSV_haralick.csv'
    process_function = fc.calc_img_haralick # Must take RGB image as input and return a list
    labels = fc.get_haralick_labels()
    num_processors = 4
    chunk_size = 100 # I have no idea what a good number for this is.

    print('Reading contents of image directory: ', image_dir)
    images = glob.glob(image_dir + '*.jpg')
    N = len(images)

    print('Processing images')
    start = time.time()
    pool = mp.Pool(processes=num_processors)
    partial_process_images = partial(process_images, process_function=process_function)

    data = np.ndarray(shape=(N, len(labels))) #  +1 for image index as order is not guaranteed
    for index, result in enumerate(pool.imap(partial_process_images, images, chunksize=chunk_size)):
        data[result[0], :] = result[1:]

    pool.close()
    pool.join()

    end = time.time()
    print('Time elapsed: ', end-start)

    print('Writing dataframe: ', output_filepath)
    df = pd.DataFrame(data=data, columns=labels)
    df.to_csv(output_filepath, index=False)

    print('Completed')
