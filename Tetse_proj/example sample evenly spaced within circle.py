# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:26:46 2019

@author: marijnhazelbag
"""

### example

import numpy as np
import matplotlib.pyplot as plt

T = [1, 10, 20, 30, 40, 50, 60]
R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]



def rtpairs(r, n):

    for i in range(len(r)):
       for j in range(n[i]):    
        yield r[i], j*(2 * np.pi / n[i])

for r, t in rtpairs(R, T):
    plt.plot(r * np.cos(t), r * np.sin(t), 'bo')
plt.show()

### real thing

import numpy as np
import matplotlib.pyplot as plt

T = [1, 8, 16, 24, 32]
R = [0.0, 2, 4, 6, 8]

#n=9
#4*(n-1)

def rtpairs(r, n):

    for i in range(len(r)):
       for j in range(n[i]):    
        yield r[i], j*(2 * np.pi / n[i])

for r, t in rtpairs(R, T):
    plt.plot(r * np.cos(t), r * np.sin(t), 'bo')
plt.show()

for r, t in rtpairs(R, T):
    x1= r * np.cos(t)
    y1= r * np.sin(t)
rtpairs(R,T)




###

range(len(R))

def rtpairs(R, N):
        R=[1, 8, 16, 24, 32]
        N = [0.0, 2, 4, 6, 8]
        r=[]
        t=[]
        for i in N:
            theta=2*np.pi/i
            t.append(theta)
            

        for j in R:
            j=j
            r.append(j)

x1[]= r*np.cos(t)
y1[]= r*np.sin(t)

    plt.plot(r*np.cos(t),r*np.sin(t), 'bo')
    plt.show()

?append

