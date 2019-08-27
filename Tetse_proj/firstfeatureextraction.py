# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:35:26 2019

@author: jem
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
image = images[0]
image=(images[0])
feature_coord, _ = haar_like_feature_coord(12, 24, 'type-2-y')
image = draw_haar_like_feature(image, 0, 0,
                               images.shape[2],
                               images.shape[1],
                               feature_coord,
                               alpha=0.1)
plt.imshow(image)

#?draw_haar_like_feature

#?haar_like_feature_coord

