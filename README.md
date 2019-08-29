# Tsetse-fly-project

Basic outline

# 1) create training data set
"""
from all images sample a space around the landmark
to obtain the samples and labels for each image
 - keep a list of the coordinates and the label
"""

# 2) Use these coordinates to find the haar like features of these coordinate pixels
""" From the coordinate samples, create a window arounf the coordinate and extract haar like features and append to a dataframe"""

# 3) Create a random forest to train the algorithm
""" Using the dataframe of haarlike features for each sample coordinate with its 'label', we train the the algorthm using these features as inputs """


    
