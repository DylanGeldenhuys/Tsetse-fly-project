# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:35:26 2019

@author: jem and Marijn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:00:38 2019

@author: marijnhazelbag
"""
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt

from dask import delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature


    #### get the images from example dataset face recognition

images = lfw_subset()
### experimenting with the HAAR like features
feature_types = ['type-2-x', 'type-2-y']
face2rec = images[0]
plt.imshow(face2rec)
ii = integral_image(face2rec)
plt.imshow(ii)
feature_coord, _ = haar_like_feature_coord(3, 3)#, 'type-3-y')
haar_feat = draw_haar_like_feature(ii, 0, 0,
                               images.shape[2],
                               images.shape[1],
                               feature_coord,
                               alpha=0.1)

plt.imshow(face2rec)
plt.imshow(haar_feat)
plt.show()
?draw_haar_like_feature
images.shape
#?haar_like_feature_coord
#rgb2gray(img)
?haar_like_feature_coord
feature_coord
np.shape(feature_coord)
 image = images[0]
 image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   feature_coord,
                                   alpha=0.001)
plt.imshow(image)

plt.scatter(feature_coord[0[0[0[0]]]],feature_coord[0[0[0[1]]]])
