# Tsetse-fly-project

Basic outline

# 1) create training data set
"""
from all images sample a space around the landmark
to obtain the samples and labels for each image
 - keep a list of the coordinates and the label
"""

# 2) Use these coordinates to find the haar like features of these coordinate pixels
""" From the coordinate samples, create a window aroundthe coordinate and extract haar like features and append to a dataframe"""

# 3) Create a random forest to train the algorithm
""" Using the dataframe of haarlike features for each sample coordinate with its 'label', we train the the algorithm using these features as inputs"""

...................................................

(Curating raw data)

Plotting_landmarks.ipynb: was used to create a dataset of all the images with their landmark plotted, this was then used to manually curate the data (taking out mismatches). It outputs into the file \imageWithLandmarks 

CuratedData.ipynb: Used the now manually curated file \imageWithLandmarks to create a new curated data set of images, outputting the coorectly labeled images into C_data

The Curated data in \C_data was then used to also make a new data file(new_textFile.csv) where all the incorect landmarks are removed.

in Plotting_landmarks.ipynb the landmarks where visualised on one image. There was still one mismatch, so the mismatch was manually removed and the process was repeated outputting the final images into \C_data2. C_data2 was then used to generate the  new_textFile.csv file.

...
(preproccesing data/ generating training set)

Only use 
Main.ipynb and RFclassifier.ipynb

Main.ipynb: generates training data files Training_data.csv from the folder C_data2 and the csv file new_textFile.txt


    
