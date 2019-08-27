# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:49:41 2019

@author: marijnhazelbag
"""

pip install -r requirements.txt

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

##The procedure to extract the Haar-like features from an image is 
# relatively simple. Firstly, a region of interest (ROI) is defined. 
# Secondly, the integral image within this ROI is computed. Finally, 
#the integral image is used to extract the features.

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
    
x= extract_feature_image(images[1], 'type-2-x')
shape(x)

?haar_like_feature
# We use a subset of CBCL dataset which is composed of 100 face 
# images and 100 non-face images. Each image has been resized to a 
# ROI of 19 by 19 pixels. We select 75 images from each group to train 
# a classifier and determine the most salient features. The remaining 
# 25 images from each class are used to assess the performance of the 
# classifier.
    
images = lfw_subset()

## check some of the images
plt.imshow(images[1])
plt.imshow(image)
plt.imshow(images[2])
##
shape(images[1])

# To speed up the example, extract the two types of features only
feature_types = ['type-2-x', 'type-2-y']

# Build a computation graph using Dask. This allows the use of multiple
# CPU cores later during the actual computation
X = delayed(extract_feature_image(img, feature_types) for img in images)
# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='threads'))
time_full_feature_comp = time() - t_start

y = np.array([1] * 100 + [0] * 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0,
                                                    stratify=y)

# Extract all possible features
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)

extract_feature_image


# A random forest classifier can be trained in order to select the 
# most salient features, specifically for face classification. 
# The idea is to determine which features are most often used by the
# ensemble of trees. By using only the most salient features in 
# subsequent steps, we can drastically speed up the computation while 
# retaining accuracy.
    
    
# Train a random forest classifier and assess its performance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance and plot the six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]

fig, axes = plt.subplots(3, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The most important features')










