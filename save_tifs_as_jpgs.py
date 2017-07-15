import glob2
from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from skimage import data, exposure, img_as_float

'''
To be run from directory with train-tif-v2
'''

try:
    os.mkdir('tif_jpg_test2')
except:
    print('Hopefully it already exists')

maxes = []
mins = []
fns = glob2.glob('test-tif-v2/*.tif')
for path in tqdm(fns):
    im = io.imread(path)
    ndvi_ = ndvi(im, 2, 3)
    ndwi = ndvi(im, 3, 1)
    rgb2 = np.zeros_like(im[:, :, :3])
    rgb3 = np.zeros_like(im[:, :, :3])
    scales = []
    inds = [im[:, :, 3], ndvi_, ndwi]
    for i in range(4):
        if i == 0:
            scaled = ((im[:, :, :3] / im[:, :, :3].mean()) * 128).astype(int)
            for i in range(3):
                rgb2[:, :, np.abs(i - 2)] = scaled[:, :, i]
        elif i == 1:
            scaled = ((inds[i - 1] / inds[i - 1].mean()) * 128).astype(int)
            # scales.append(((scaled/scaled.max())*255).astype(int))
            rgb3[:, :, i - 1] = scaled
        else:
            scaled = np.clip((inds[i - 1] + 0.5) * 128, 0, 255)
            scales.append(scaled)
            rgb3[:, :, i - 1] = scaled
    rgb2 = np.clip(rgb2, 0, 255).astype(np.uint8)
    rgb3 = np.clip(rgb3, 0, 255).astype(np.uint8)
    io.imsave('tif_jpg_test/rgb_%s' % path.split('\\')[1].replace('tif', 'jpg'), rgb2, quality=92)
    io.imsave('tif_jpg_test/ind_%s' % path.split('\\')[1].replace('tif', 'jpg'), rgb3, quality=92)
