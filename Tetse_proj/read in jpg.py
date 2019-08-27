# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:09:43 2019

@author: marijnhazelbag
"""

import os
os.chdir("C:\\Users\jeremyb\\Downloads\\TseTse Project\\Tsetse Coordinates\\2")
    ## set directory relative to location of this script, or just relative to "current" directory so we can move around as we process all the data
    

### it seems that the refined_data.txt from file #1 does not match the pictures
### included in that file...

"""
Import coordinates
"""


import pandas as pd
df = pd.read_fwf('refined_data.txt',header=None)

new = df.iloc[:,4].str.split(" ", n = 23, expand = True) 

df= df.drop(labels=4, axis=1)

result = pd.concat([df, new], axis=1, sort=False)


"""
Import a single jpg
"""

from PIL import Image
jpgfile = Image.open("A002 - 20170126_195145.jpg")

print(jpgfile.bits, jpgfile.size, jpgfile.format)

"""
Below plots the jpg
"""

%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('A002 - 20170126_195145.jpg')
imgplot = plt.imshow(img, zorder=1)
plt.show()

### Try to overlay the landmark points



# put a blue dot at (10, 20)

plt.imshow(img)
plt.scatter([402.523], [301.503763])
plt.scatter([558.891884541], [320.74913377])
plt.scatter([1117.00761996], [551.69357601])
plt.scatter([1203.6117858], [623.86371421])
plt.scatter([1222.85715598], [657.543112037])
plt.scatter([862.006464982], [633.486399304])
plt.scatter([722.477531128], [700.845194957])
plt.scatter([647.901721655], [467.495081443])
plt.scatter([616.627995101], [426.598669797])
plt.scatter([337.570127394], [404.947628337])
plt.scatter([260.588646648], [842.779800084])



a= [pd.to_numeric(new.iloc[0,5])]
b= [pd.to_numeric(new.iloc[0,6])]
plt.scatter(float(new.iloc[0,5]), float(new.iloc[0,6]))
plt.scatter(a, b)

plt.show()


# put a red dot, size 40, at 2 locations:
plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)

plt.show()



