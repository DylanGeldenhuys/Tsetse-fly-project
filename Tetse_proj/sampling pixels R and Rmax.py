# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:10:58 2019

@author: marijnhazelbag
"""
import numpy as np
R= 9 ### with R 9 the second plot is quite hard to tell whether it works 
Rmax= 300

npospix = 100
nnegpix = 2*npospix

a = np.random.uniform(0,1,npospix) * 2 * np.pi
r = R * np.sqrt(np.random.uniform(0,1,npospix))

## If you need it in Cartesian coordinates
xwithin = r * np.cos(a)
ywithin = r * np.sin(a)

plt.plot(xwithin, ywithin, 'ro')


     
# import the math module  
import math  
    
### (Python real deal) 
## generate random values
outer_radius =Rmax*Rmax
inner_radius = R*R
rho= np.sqrt(np.random.uniform(inner_radius, 
                             outer_radius, size=nnegpix))

theta=  np.random.uniform( 0, 2*np.pi, nnegpix)
xhoop = rho * np.cos(theta)
yhoop = rho * np.sin(theta)
plt.plot(xhoop, yhoop, 'ro')

#?np.sin

