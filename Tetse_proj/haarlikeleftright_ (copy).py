{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Created on Fri Jul 19 16:44:55 2019\n",
    "\n",
    "@author: jem\n",
    "\"\"\"\n",
    "\n",
    "#%matplotlib inline\n",
    "\n",
    "# Exploratory left/right classification using Haar-like feature descriptor\n",
    "\n",
    "\n",
    "\n",
    "# Face classification using Haar-like feature descripton\n",
    "import pandas as pd\n",
    "import sys\n",
    "from time import time\n",
    "\n",
    "### for ROC curve metrics\n",
    "from sklearn import metrics\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from dask import delayed\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "from skimage.data import lfw_subset\n",
    "from skimage.transform import integral_image\n",
    "from skimage.feature import haar_like_feature\n",
    "from skimage.feature import haar_like_feature_coord\n",
    "from skimage.feature import draw_haar_like_feature\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def rgb2gray(rgb):\n",
    "    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])\n",
    "\n",
    "\n",
    "import os\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "\n",
    "#directories\n",
    "main = \"/home/dylan/Projects/Tetse_proj/\"\n",
    "data_2 = \"/home/dylan/Projects/Tetse_proj/2/\"\n",
    "data_2_small = \"/home/dylan/Projects/Tetse_proj/2/2_small/\"\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Read in Label data (result), set wd, test image read-into numpy array\n",
    "\n",
    "\n",
    "df = pd.read_fwf(data_2 + 'refined_data.txt',header=None)\n",
    "#?pd.read_fwf\n",
    "# which function\n",
    "which = lambda lst:list(np.where(lst)[0])\n",
    "\n",
    "new = df.iloc[:,4].str.split(\" \", n = 23, expand = True)\n",
    "\n",
    "df= df.drop(labels=4, axis=1)\n",
    "\n",
    "result = pd.concat([df, new], axis=1, sort=False)\n",
    "\n",
    "# column numbers were messed up, make them go according to range\n",
    "result.columns = range(result.shape[1])\n",
    "\n",
    "result.reset_index(inplace=True, drop=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#read in all pictures based on the names in refined_data.txt, \n",
    "#resize them to \"thumbnails\"\n",
    "#and save them to a hard-coded (for now) folder called 2_small using the same names as the originals\n",
    "# (so that the labelling doesn't need adjustment)\n",
    "# we also manually copied refined_data.txt into the 2_small folder for now\n",
    "#print(os.getcwd())\n",
    "\n",
    "for x in result.index:\n",
    "    imgname=\" \".join(result.loc[x,[0,1,2]])\n",
    "    #file, ext = os.path.splitext(imgname)\n",
    "    im = Image.open(data_2 + imgname)\n",
    "    im.thumbnail([22,22])    #### CHANGE HERE!!!!  !!!!!! ######\n",
    "#    os.chdir(\"Projects/Tetse_proj/2/2_small\")   #### \n",
    "    im.save(data_2_small + imgname)\n",
    "##    print(np.as.ndarray(im))\n",
    "#    os.chdir(\"..\\\\\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "# make all images greyscale\n",
    "for x in result.index:\n",
    "    imgname=\" \".join(result.loc[x,[0,1,2]])\n",
    "#    file, ext = os.path.splitext(imgname)\n",
    "    img = mpimg.imread(data_2_small + imgname)     \n",
    "    I = rgb2gray(img)  #make it greyscale\n",
    "    I8 = (((I-I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8) # the next few lines help us save the images as true greyscale (ie. each image is 2 dimensional)\n",
    "    greyimg = Image.fromarray(I8)\n",
    "    rescaledwidth, rescaledheight = np.shape(greyimg)\n",
    "    greyimg.save(data_2_small  + 'bw/' + imgname)\n",
    "    #print(np.shape(mpimg.imread(imgname)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fe16aa85320>"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#### creates fullset containing all the pictures\n",
    "#### access the first picture by fullset[1] \n",
    "\n",
    "# get rescaled width and height\n",
    "imgname=\" \".join(result.loc[5,[0,1,2]])\n",
    "img = mpimg.imread(data_2 + imgname) \n",
    "#print(np.shape(img))\n",
    "rescaledwidth, rescaledheight, pix = np.shape(img)\n",
    "#print(type(rescaledwidth), type(rescaledheight), type(pix))\n",
    " \n",
    "fullset = {}\n",
    "\n",
    "\n",
    "for i in result.index:\n",
    "    fullset[i] = mpimg.imread(data_2_small  + \" \".join(result.loc[i,[0,1,2]]))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fe16a1e1828>"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "### create outcome vector y as in the example\n",
    "### y contains a vector of TRUE and FALSE, TRUE indicates left wings\n",
    "y= result.loc[:,0]==\"A002\"\n",
    "y=np.asarray(y)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
