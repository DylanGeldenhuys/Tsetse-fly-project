# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:35:21 2019

@author: Dylan Jeremy and Marijn
"""


import os
import pandas as pd

print(os.getcwd())
os.chdir(".\\Tsetse Coordinates")
print(os.getcwd())
print(os.listdir(os.getcwd()))

for filename in os.listdir(os.getcwd()):
    os.chdir(filename)    
    df = pd.read_csv('refined_data.txt',sep= " ",header=None)
    
    #do stuff
    print(df.head(3))
    os.chdir("../")

#os.chdir(".\\1")


#df = pd.read_csv('refined_data.txt',sep= " ",header=None)

#print(df.head(3))


#new = df.iloc[:,4].str.split(" ", n = 23, expand = True) 

#df= df.drop(labels=4, axis=1)

#result = pd.concat([df, new], axis=1, sort=False)
