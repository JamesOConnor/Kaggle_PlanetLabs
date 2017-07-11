import glob2
from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

'''
To be run from directory with train-tif-v2
'''

try:
	os.mkdir('train_tif_reshape')
except:
	print('Hopefully it already exists')

fns = glob2.glob('train-tif-v2/*.tif')
for path in tqdm(fns):
	im = io.imread(path)
	img2 = get_rgb(im, [3, 2, 1])
	rgb = get_rgb(im, [2, 1, 0]) 

	rescaleIMG = np.reshape(rgb, (-1, 1))
	scaler = MinMaxScaler(feature_range=(0, 255))
	rescaleIMG = scaler.fit_transform(rescaleIMG) # .astype(np.float32)
	rgb_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)

	vi2 = (img2[:, :, 0] - img2[:, :, 1]) / (img2[:, :, 0] + img2[:, :, 1]) # (NIR - RED) / (NIR + RED)
	vi3 = (img2[:, :, 2] - img2[:, :, 0]) / (img2[:, :, 2] + img2[:, :, 0]) # (GREEN - NIR) / (GREEN + NIR)
	
	if vi2.min() < 0:
		vi2 = vi2 - vi2.min()
	if vi3.min() < 0:
		vi3 = vi3 - vi3.min()
		
	fac1 = 1/(vi2.max()/255)
	fac2 = 1/(vi3.max()/255)
	fac3 = 1/(img2[:,:,0].max()/255)
	
	vi2_scale = (vi2*fac1).astype(np.uint8)
	vi3_scale = (vi3*fac2).astype(np.uint8)
	nir = (img2[:,:,0]*fac3).astype(np.uint8)
	
	im2 = np.dstack((vi2_scale, vi3_scale, nir))
	
	io.imsave('train_tif_reshape/rgb_%s'%path.split('\\')[1].replace('tif', 'jpg'), rgb_scaled)
	io.imsave('train_tif_reshape/ind_%s'%path.split('\\')[1].replace('tif', 'jpg'), im2)