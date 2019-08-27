# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:06:02 2019

@author: marijnhazelbag
"""


import os
import numpy as np

print(os.getcwd())
os.chdir(".\\Tsetse Coordinates\\2")

reshere = 1
img=mpimg.imread(" ".join(resultLeft.loc[reshere,[0,1,2]]))

### you need to run read data before running below

plt.imshow(img)
plt.scatter(float(resultLeft.loc[resultLeft.index[reshere], 15]), 
            float(resultLeft.loc[resultLeft.index[reshere],16]),
            s=5)


plt.imshow(img)
plt.xlim(600, 800)
plt.ylim(600, 800)
plt.scatter(float(resultLeft.loc[resultLeft.index[reshere], 15]), 
            float(resultLeft.loc[resultLeft.index[reshere],16]),
            s=10)

### this needs to be run before running further below!!!
x1landmark= float(resultLeft.loc[resultLeft.index[reshere], 15])
y1landmark= float(resultLeft.loc[resultLeft.index[reshere],16])

plt.imshow(img)
for i in range(len(xhoop)):
    plt.plot(x1landmark + xwithin[i], y1landmark + ywithin[i], 'ro')
    plt.plot(x1landmark + xhoop[i], y1landmark + yhoop[i], 'bo')
plt.show()
    

### only within
plt.imshow(img)
for i in range(len(xwithin)):
    plt.plot(x1landmark + xwithin[i], y1landmark + ywithin[i], 'ro')
plt.show()
    
### only the hoop
plt.imshow(img)
for i in range(len(xhoop)):
    plt.plot(x1landmark + xhoop[i], y1landmark + yhoop[i], 'bo')
plt.show()

?scatter
?imshow


plt.imshow(img)
#plt.xlim(600, 800)
#plt.ylim(600, 800)
for i in range(len(xhoop)):
    plt.plot(x1landmark + xwithin[i], y1landmark + ywithin[i], 'ro')
for i in range(len(xwithin)):
    plt.plot(x1landmark + xhoop[i], y1landmark + yhoop[i], 'bo')


for picturenumber in range(np.shape(resultLeft)[0]):
    plt.scatter(float(resultLeft.loc[picturenumber,15]), float(resultLeft.loc[picturenumber,16]),)
plt.show()
    
#for picturenumber in range(np.shape(resultLeft)[0]):
#    plt.scatter(float(resultLeft.loc[picturenumber,2*coordinatenumber + 3 ]), float(resultLeft.loc[picturenumber,2*coordinatenumber + 4]),)
