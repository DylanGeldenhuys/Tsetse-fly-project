# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:00:38 2019

@author: marijnhazelbag
"""
#############3

from array import array

def extract_feature_image(img, feature_coord):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

img1 =mpimg.imread(" ".join(resultLeft.loc[resultLeft.index[1],[0,1,2]]))
plt.imshow(img1)

### this needs to be run before running further below!!!
x1landmark= float(resultLeft.loc[resultLeft.index[1], 15])
y1landmark= float(resultLeft.loc[resultLeft.index[1],16])

### create coordinates for which we are going to create the HAAR like features
x1LMcoord = numpy.empty(len(xwithin), dtype=object)
y1LMcoord = numpy.empty(len(xwithin), dtype=object)
plt.imshow(img1)
for i in range(len(xhoop)):
    plt.plot(x1landmark + xwithin[i], y1landmark + ywithin[i], 'ro')
    x1LMcoord[i]= x1landmark + xwithin[i]
    y1LMcoord[i]= y1landmark + ywithin[i]
plt.show()

### round to integers
x1LMcoord= np.around(x1LMcoord.astype(np.double))
y1LMcoord= np.around(y1LMcoord.astype(np.double))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
img1 = imutils.resize(img)

### make actual coordinates (cset)

coordinates = zip(x1LMcoord, y1LMcoord)
coordinateSet = set(coordinates)
cset=  np.tupple(coordinateSet)

### use coordinates to extract HAAR like features from image at coordinates


shape(img1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('image.png')     
gray = rgb2gray(img1)    
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

plt.imshow(gray)
shape(gray)

?haar_like_feature


extract_feature_image(gray, feature_coord=cset)



image = draw_haar_like_feature(gray, 0, 0,
                               600,
                               600,
                               'type-2-x')


### experimenting with the HAAR like features
feature_types = ['type-2-x', 'type-2-y']
image = images[0]
image=(images[0])
feature_coord, _ = haar_like_feature_coord(12, 24, 'type-2-y')
image = draw_haar_like_feature(image, 0, 0,
                               images.shape[2],
                               images.shape[1],
                               feature_coord,
                               alpha=0.1)
plt.imshow(image)

?draw_haar_like_feature

?haar_like_feature_coord

import numpy as np
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
feature_coord, _ = haar_like_feature_coord(2, 2, 'type-3')
image = draw_haar_like_feature(np.zeros((2, 2)),
                               0, 0, 2, 2,
                               feature_coord,
                               max_n_features=1)


feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')
image = draw_haar_like_feature(np.zeros((2, 2)),
                               0, 0, 2, 2,
                               feature_coord,
                               max_n_features=1,
                               alpha=0.5)
plt.imshow(image)

