# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:09:43 2019

@author: marijnhazelbag
"""

### it seems that the refined_data.txt from file #1 does not match the pictures
### included in that file...

"""
Import coordinates
"""

import os
import numpy as np

print(os.getcwd())
os.chdir(".\\Tsetse Coordinates\\2")

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

"""
Import a single jpg
"""

from PIL import Image
jpgfile = Image.open("A002 - 20170126_195145.jpg")

print(jpgfile.bits, jpgfile.size, jpgfile.format)

"""
Below plots the jpg
"""

#%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('A002 - 20170126_195145.jpg')
imgplot = plt.imshow(img, zorder=1)
plt.show()

### Try to overlay the landmark points


plt.imshow(img)
plt.scatter(float(result.loc[0, 15]), float(result.loc[0,16]))



# create for loop to loop over images and plot the dots
#len(result.index)
#resultLeft= result[result.loc[:,[1]] == 'A002',:]

plt.imshow(img)
for x in result.index:
  plt.scatter(float(result.loc[x, 15]), float(result.loc[x,16]))

# plot the separete pictures to check whether there are any mistakes
  
img=mpimg.imread('A002 - 20170126_195145.jpg')
imgplot = plt.imshow(img, zorder=1)
plt.show()

plt.imshow(img)

result.loc[0,[0]]

result.reset_index(inplace=True, drop=True)
result.columns = range(result.shape[1])

result.loc[0,[0,1,2]]
" ".join(result.loc[0,[0,1,2]])

for x in result.index:
    img=mpimg.imread(" ".join(result.loc[x,[0,1,2]]))
    plt.imshow(img)

onlyA2 = result.loc[:,[0]]
which(onlyA2 == 'A002')

resultRight= result.loc[which(onlyA2 == 'A002'),:]

result.loc[0,[0,1,2]]



### plot only for right wings
from matplotlib.backends.backend_pdf import PdfPages

plt.imshow(img)
with PdfPages('multipage_pdf_right.pdf') as pdf:
    for x in resultRight.index:
      img=mpimg.imread(" ".join(resultRight.loc[x,[0,1,2]]))
      plt.imshow(img)
      plt.scatter(float(resultRight.loc[x, 15]), float(resultRight.loc[x,16]))
      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()


### For the Left wings

onlyA3 = result.loc[:,[0]]

resultLeft= result.loc[which(onlyA3 == 'A003'),:]

img=mpimg.imread(" ".join(resultLeft.loc[resultLeft.index[1],[0,1,2]]))

resultLeft.reset_index(inplace=True, drop=True)

plt.imshow(img)
for x in resultLeft.index:
  plt.scatter(float(resultLeft.loc[x, 15]), float(resultLeft.loc[x,16]))

plt.imshow(img)
with PdfPages('multipage_pdf_Left.pdf') as pdf:
    for x in resultLeft.index:
      img=mpimg.imread(" ".join(resultLeft.loc[x,[0,1,2]]))
      plt.imshow(img)
      plt.scatter(float(resultLeft.loc[x, 15]), float(resultLeft.loc[x,16]))
      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()

## look at wronfully right wing in our left wing data 


reshere = 27
img=mpimg.imread('A002 - 20170126_195145.jpg')
img=mpimg.imread(" ".join(resultLeft.loc[reshere,[0,1,2]]))
plt.imshow(img)

for coordinatenumber in range(11):
    print(2*coordinatenumber + 3 )
    print(2*coordinatenumber + 4 )
    plt.scatter(float(resultLeft.loc[reshere,2*coordinatenumber + 3 ]), float(resultLeft.loc[reshere,2*coordinatenumber+4]),)


## create plot with all points plotted for each wing
    
    
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('multipage_pdf_left_allpoints.pdf') as pdf:
    for x in resultLeft.index:
      img=mpimg.imread(" ".join(resultLeft.loc[x,[0,1,2]]))
      plt.imshow(img)
      for coordinatenumber in range(11):
          plt.scatter(float(resultLeft.loc[x,2*coordinatenumber + 3 ]), float(resultLeft.loc[x,2*coordinatenumber + 4]),)
    #  plt.scatter(float(resultRight.loc[x, 15]), float(resultRight.loc[x,16]))
      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()


##what's with coordinate 24???
      
with PdfPages('multipage_pdf_left_lastpoint.pdf') as pdf:
    for x in resultLeft.index:
      img=mpimg.imread(" ".join(resultLeft.loc[x,[0,1,2]]))
      plt.imshow(img)
      plt.scatter(float(resultLeft.loc[x,2*10 + 3 ]), float(resultLeft.loc[x,2*10 + 4]),)
    #  plt.scatter(float(resultRight.loc[x, 15]), float(resultRight.loc[x,16]))
      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()

      
reshere = 27
img=mpimg.imread('A002 - 20170126_195145.jpg')
img=mpimg.imread(" ".join(resultLeft.loc[reshere,[0,1,2]]))
plt.imshow(img)

for coordinatenumber in range(11):
    print(2*coordinatenumber + 3 )
    print(2*coordinatenumber + 4 )
    plt.scatter(float(resultLeft.loc[reshere,2*coordinatenumber + 3 ]), float(resultLeft.loc[reshere,2*coordinatenumber+4]),)


?plt.scatter