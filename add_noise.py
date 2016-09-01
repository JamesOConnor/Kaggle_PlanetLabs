import skimage.util as noi
import cv2
import numpy as np
import sys

infile=sys.argv[1]
snr=float(sys.argv[2])

x=cv2.imread(infile).astype(float)

x = x - np.min(x);
x = x / np.max(x);

v = np.var(x) / (10**(snr/10));
x_noise=(noi.random_noise(x, mode='gaussian', mean=0, var=v)*255).astype(np.uint8)
cv2.imwrite('%s'%(infile.replace('.JPG', '_noisy.JPG')), x_noise)
#
