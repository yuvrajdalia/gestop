import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import os
import cv2
from skimage import exposure
path = 'G:\projects\offnote\\virtualboard\MediaPipePyTorch\dataset_curvy'
for count, filename in enumerate(os.listdir("dataset_curvy")):
    print(filename)
    img = imread(path+'/'+filename) / 255
    #plt.imshow(img)
    #plt.show()
    newfnamex=filename[:-4]
    newfnamex=newfnamex+'_n.jpg'
    #newfname = rotate(img, angle=180)
    #newfname = exposure.adjust_gamma(img, gamma=0.4, gain=0.9)
    newfname = random_noise(img, var=0.1**2)
    plt.imshow(newfname)
    plt.imsave(newfnamex,newfname )
    #plt.imshow(newfname)
    #plt.show()
    #plt.savefig(newfnamex)

    