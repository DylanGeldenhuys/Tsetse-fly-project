# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:02:58 2019

@author: marijnhazelbag
"""

plt.imshow(images[5])

from matplotlib.backends.backend_pdf import PdfPages

forxhere1= np.arange(0, 20, 1)
forxhere2= np.arange(0, 40000, 1000)

forxhere= np.concatenate((forxhere1,forxhere2), axis=None)

with PdfPages('oneimage_all_HAAR-like.pdf') as pdf:
    for x in forxhere:
      image= images[5]
      image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[x]])
      plt.imshow(image)
    #  plt.scatter(float(resultRight.loc[x, 15]), float(resultRight.loc[x,16]))
      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()

      
feature_coord[250]      
      