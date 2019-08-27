# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:44:55 2019

@author: jem
"""

#%matplotlib inline

# Exploratory left/right classification using Haar-like feature descriptor



# Face classification using Haar-like feature descripton



import sys
from time import time

### for ROC curve metrics
from sklearn import metrics

import numpy as np
import pandas as pd
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#Read in Label data (result), set wd, test image read-into numpy array

import os
import numpy as np
from PIL import Image

#print(os.getcwd())
os.chdir("C:\\Users\\marijnhazelbag\\Documents\\TseTse project\\Jeremy 19_Jul_2019\\2") # this changes the working directory 

import pandas as pd
df = pd.read_fwf('refined_data.txt',header=None)
#?pd.read_fwf
# which function
which = lambda lst:list(np.where(lst)[0])

new = df.iloc[:,4].str.split(" ", n = 23, expand = True)

df= df.drop(labels=4, axis=1)

result = pd.concat([df, new], axis=1, sort=False)

# column numbers were messed up, make them go according to range
result.columns = range(result.shape[1])

result.reset_index(inplace=True, drop=True)


img=mpimg.imread(" ".join(result.loc[5,[0,1,2]]))
#    plt.imshow(img)
plt.imshow(img)
plt.imshow(rgb2gray(img))
print(np.shape(rgb2gray(img)))

#test script to thumbnail one image
file, ext = os.path.splitext("A002 - 20170126_162639.jpg")
im = Image.open("A002 - 20170126_162639.jpg")
im.thumbnail([5,5])
im.save(file + "-thumbnail2.jpg", "JPEG")


#read in all pictures based on the names in refined_data.txt, 
#resize them to "thumbnails"
#and save them to a hard-coded (for now) folder called 2_small using the same names as the originals
# (so that the labelling doesn't need adjustment)
# we also manually copied refined_data.txt into the 2_small folder for now

df = pd.read_fwf('refined_data.txt',header=None)
#?pd.read_fwf
# which function
which = lambda lst:list(np.where(lst)[0])

new = df.iloc[:,4].str.split(" ", n = 23, expand = True)

df= df.drop(labels=4, axis=1)

result = pd.concat([df, new], axis=1, sort=False)

# column numbers were messed up, make them go according to range
result.columns = range(result.shape[1])

result.reset_index(inplace=True, drop=True)


#plt.imshow(img)

#result.loc[0,[0]]

result.reset_index(inplace=True, drop=True)
result.columns = range(result.shape[1])

result.loc[0,[0,1,2]]
" ".join(result.loc[0,[0,1,2]])
os.chdir("C:\\Users\\marijnhazelbag\\Documents\\TseTse project\\Jeremy 19_Jul_2019\\2")
for x in result.index:
    imgname=" ".join(result.loc[x,[0,1,2]])
#    file, ext = os.path.splitext(imgname)
    im = Image.open(imgname)
    im.thumbnail([22,22])    #### CHANGE HERE!!!!  !!!!!! ######
    os.chdir(".\\2_small")   #### 
    im.save(imgname)         #####
#    print(np.as.ndarray(im))
    os.chdir("..\\")

# make all images greyscale

os.chdir("C:\\Users\\marijnhazelbag\\Documents\\TseTse project\\Jeremy 19_Jul_2019\\2\\2_small")

for x in result.index:
    imgname=" ".join(result.loc[x,[0,1,2]])
#    file, ext = os.path.splitext(imgname)
    img = mpimg.imread(imgname)     
    I = rgb2gray(img)  #make it greyscale
    I8 = (((I-I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8) # the next few lines help us save the images as true greyscale (ie. each image is 2 dimensional)
    os.chdir("bw")
    greyimg = Image.fromarray(I8)
    rescaledwidth, rescaledheight = np.shape(greyimg)
    greyimg.save(imgname)
    #print(np.shape(mpimg.imread(imgname)))
    os.chdir("..\\")
    

imgname=" ".join(result.loc[5,[0,1,2]])
#    file, ext = os.path.splitext(imgname)
os.chdir("bw")
img = mpimg.imread(imgname)  
#    plt.imshow(img)
plt.imshow(img)
rescaledwidth, rescaledheight = np.shape(img)
os.chdir("..\\")

#### creates fullset containing all the pictures
#### access the first picture by fullset[1] 
os.chdir("C:\\Users\\marijnhazelbag\\Documents\\TseTse project\\Jeremy 19_Jul_2019\\2\\2_small\\bw")
fullset = np.ndarray(shape=[464,rescaledwidth,rescaledheight])
print(np.shape(fullset[1]))
for i in range(464):
    fullset[i] = mpimg.imread(" ".join(result.loc[i,[0,1,2]]))
os.chdir("..\\")

print(fullset[1].max())

plt.imshow(fullset[5]/255,vmin=0,vmax=1)

### create outcome vector y as in the example
### y contains a vector of TRUE and FALSE, TRUE indicates left wings
y= result.loc[:,0]=="A002"
y=np.asarray(y)
type(y)

#os.chdir("bw")
#print(np.shape(mpimg.imread("A002 - 20170126_195145.jpg")))
#os.chdir("..\\")
print(rescaledwidth,rescaledheight)
print(np.shape(result))
print(np.shape(y))

#make them grey
#make them small


np.shape(fullset[5])

#np.shape()

#for x in range(461):
#    img=mpimg.imread(" ".join(result.loc[x,[0,1,2]]))
#    #print(np.shape(img))
#    fullset[x] = img
#fullset[5]

plt.imshow(fullset[5].astype(np.uint8))

np.shape(result)


##### from example to see how to do it
#type(result)
#images = lfw_subset()
#type(images)
#print(np.shape(images))
#images
#images = lfw_subset()
#print(type(images))
#print(np.shape(images))


print(np.shape(fullset))

print(lfw_subset()[1].max())
#print( lfw_subset()[1])
#print(fullset[1]/255.9)

images = fullset/255.9 # rescaling to 1 so we avoid getting Nan in "X"
#images = lfw_subset()

#### new to avoid error perhaps, doesn't work
#images= fullset/(255.9*100000)


images[1]

plt.imshow(images[200])
images[5]

print(type(images))





@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

#?haar_like_feature

# To speed up the example, extract the two types of features only
feature_types = ['type-4', 'type-2-x', 'type-2-y']

# Build a computation graph using Dask. This allows the use of multiple
# CPU cores later during the actual computation
X = delayed(extract_feature_image(img, feature_types) for img in images)
print(X.max())




X[1]

#?extract_feature_image

# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='threads'))
time_full_feature_comp = time() - t_start



##### what to do with the LARGE numbers (e.g. 202.34e+303)
def make_inf_small(value):
    if abs(value) > 10000:
        value = np.sign(value)*100
    return value

#X2 = np.zeros(shape=np.shape(X))
for a in range(np.shape(X)[0]):
    for b in range(np.shape(X)[1]):
        X[a][b] = make_inf_small(X[a][b])


#def remove_exponent(value):
#    """
#       >>>(Decimal('5E+3'))
#       Decimal('5000.00000000')
#    """
#    decimal_places = 8
#    max_digits = 16
#
#    if isinstance(value, decimal.Decimal):
#        context = decimal.getcontext().copy()
#        context.prec = max_digits
#        return "{0:f}".format(value.quantize(decimal.Decimal(".1") ** decimal_places, context=context))
#    else:
#        return "%.*f" % (decimal_places, value)

###############################################################################
####
##### working here!!! !!



#### remove the NA's
X= np.nan_to_num(X)

#X= remove_exponent(X)

#y = np.array([1] * 100 + [0] * 100) #this makes a vector of 1s and 0s

### create outcome vector y as in the example
### y contains a vector of TRUE and FALSE, TRUE indicates left wings
y= result.loc[:,0]=="A002"
y=np.asarray(y)
type(y)

## we defined y up above
print('almost')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0,
                                                    stratify=y)
print('set made')
# Extract all possible features
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)
print('features_extracted')

print(X.max())



#print(X_train)
#for i in range(465):
 #   print(X_train[i]==fullset[i])



# Train a random forest classifier and assess its performance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance and plot the six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]

fig, axes = plt.subplots(3, 4)
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


cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
cdf_feature_importances /= np.max(cdf_feature_importances)
### 0.9 is increased from 0.7 (gives better classification)
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 1)
print(('{} features, or {}%, account for 70% of branch points in the '
       'random forest.').format(sig_feature_count, sig_feature_percent))

# Select the determined number of most informative features
feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
# Note: it is also possible to select the features directly from the matrix X,
# but we would like to emphasize the usage of `feature_coord` and `feature_type`
# to recompute a subset of desired features.

#?extract_feature_image

# Build the computational graph using Dask
X = delayed(extract_feature_image(img, feature_type_sel, feature_coord_sel)
            for img in images)
# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='threads'))
time_subs_feature_comp = time() - t_start

#y = np.array([1] * 100 + [0] * 100)
y= result.loc[:,0]=="A002"
y=np.asarray(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0,
                                                    stratify=y)

### this was inserted because gave error wrt max_features,
### so changed that to the # features extracted in summary above
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=5, n_jobs=-1, random_state=0)


t_start = time()
clf.fit(X_train, y_train)
time_subs_train = time() - t_start

auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

summary = (('Computing the full feature set took {:.3f}s, plus {:.3f}s '
            'training, for an AUC of {:.2f}. Computing the restricted '
            'feature set took {:.3f}s, plus {:.3f}s training, '
            'for an AUC of {:.2f}.')
           .format(time_full_feature_comp, time_full_train,
                   auc_full_features, time_subs_feature_comp,
                   time_subs_train, auc_subs_features))

print(summary)
plt.show()

y_pred= clf.predict_proba(X_test)[:, 1]
d = {'Left wing': y_test, 'Prob left based on classifier': y_pred}
df2 = pd.DataFrame(data=d)
df2

### check which ones unequal
np.equal(y_test==1, y_pred<0.5)

np.sum(np.equal(y_test==1, y_pred<0.5))

np.sum(np.equal(y_test==1, y_pred<0.6))
np.sum(np.equal(y_test==1, y_pred<0.7))
np.sum(np.equal(y_test==1, y_pred<0.55))

fpr, tpr, thresh = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
auc = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
plt.plot(fpr,tpr)

diff =tpr-fpr
d2= {'fal_pos':fpr, 'true_pos':tpr, 'threshold':thresh, 'tpr-fpr': diff}
df3 = pd.DataFrame(data=d2)

from pprint import pprint
opt = np.get_printoptions()
np.set_printoptions(threshold=sys.maxsize)
list(df3)
pd.set_option('display.max_rows', 1000)
df3

np.set_printoptions(**opt)


(y_test==1) and (y_pred<0.5)