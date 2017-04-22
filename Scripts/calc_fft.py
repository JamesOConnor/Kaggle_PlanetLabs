import cv2
import numpy as np

def calc_fft():
    '''
    Calculates and returns the power spectra of each image band, see ref Torralba, 2003 - Statistics of natural image
    categories
    '''
    mean_fft_green = np.zeros_like(train[1:, 0])
    mean_fft_red = np.zeros_like(train[1:, 0])
    mean_fft_blue = np.zeros_like(train[1:, 0])
    std_fft_green = np.zeros_like(train[1:, 0])
    std_fft_red = np.zeros_like(train[1:, 0])
    std_fft_blue = np.zeros_like(train[1:, 0])

    for i in range(len(train[1:])):
        if i % 50 == 0:
            print i
        im = cv2.imread('train-jpg/train_%s.jpg' % (int(i)))
        greenfft = abs(np.fft.fftshift(np.fft.fft2(im[:, :, 1])))
        redfft = abs(np.fft.fftshift(np.fft.fft2(im[:, :, 0])))
        bluefft = abs(np.fft.fftshift(np.fft.fft2(im[:, :, 2])))
        mean_fft_green[i] = greenfft.mean()
        mean_fft_red[i] = redfft.mean()
        mean_fft_blue[i] = bluefft.mean()
        std_fft_green[i] = greenfft.std()
        std_fft_red[i] = redfft.std()
        std_fft_blue[i] = bluefft.std()
    return np.r_['1,2,0', mean_fft_green, mean_fft_red, mean_fft_blue, std_fft_green, std_fft_red, std_fft_blue]