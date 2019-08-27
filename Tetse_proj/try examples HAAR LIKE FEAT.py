# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:32:47 2019

@author: marijnhazelbag
"""



images = lfw_subset()
### experimenting with the HAAR like features
feature_types = ['type-2-x', 'type-2-y']
image = images[0]
image=(images[0])
feature_coord, _ = haar_like_feature_coord(12, 24, 'type-2-y')
image = draw_haar_like_feature(image, 0, 0,
                               images.shape[2],
                               images.shape[1],
                               feature_coord,
                               alpha=0.1)
plt.imshow(image)

?draw_haar_like_feature

?haar_like_feature_coord










import numpy as np
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
img = np.ones((5, 5), dtype=np.uint8)
img_ii = integral_image(img)
feature = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-x')
feature
array([-1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1,
       -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -1, -2, -3, -1,
       -2, -1, -2, -1, -2, -1, -1, -1])


from skimage.feature import haar_like_feature_coord
feature_coord, feature_type = zip(
    *[haar_like_feature_coord(5, 5, feat_t)
      for feat_t in ('type-2-x', 'type-3-x')])
# only select one feature over two
feature_coord = np.concatenate([x[::2] for x in feature_coord])
feature_type = np.concatenate([x[::2] for x in feature_type])
feature = haar_like_feature(img_ii, 0, 0, 5, 5,
                            feature_type=feature_type,
                            feature_coord=feature_coord)
plt.imshow(img)
plt.imshow(img_ii)

?haar_like_feature

feature
array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0, -1, -3, -1, -3, -1, -3, -1, -3, -1,
       -3, -1, -3, -1, -3, -2, -1, -3, -2, -2, -2, -1])


array([list([[(0, 0), (0, 0)], [(0, 1), (0, 1)]]),
       list([[(0, 0), (1, 0)], [(0, 1), (1, 1)]]),
       list([[(0, 0), (2, 0)], [(0, 1), (2, 1)]]),
       list([[(0, 0), (3, 0)], [(0, 1), (3, 1)]]),
       list([[(0, 1), (0, 1)], [(0, 2), (0, 2)]]),
       list([[(0, 1), (1, 1)], [(0, 2), (1, 2)]]),
       list([[(0, 1), (2, 1)], [(0, 2), (2, 2)]]),
       list([[(0, 1), (3, 1)], [(0, 2), (3, 2)]]),
       list([[(0, 2), (0, 2)], [(0, 3), (0, 3)]]),
       list([[(0, 2), (2, 2)], [(0, 3), (2, 3)]]),
       list([[(0, 3), (0, 3)], [(0, 4), (0, 4)]]),
       list([[(0, 3), (2, 3)], [(0, 4), (2, 4)]]),
       list([[(1, 0), (1, 0)], [(1, 1), (1, 1)]]),
       list([[(1, 0), (2, 0)], [(1, 1), (2, 1)]]),
       list([[(1, 0), (3, 0)], [(1, 1), (3, 1)]]),
       list([[(1, 0), (4, 0)], [(1, 1), (4, 1)]]),
       list([[(1, 1), (1, 1)], [(1, 2), (1, 2)]]),
       list([[(1, 1), (2, 1)], [(1, 2), (2, 2)]]),
       list([[(1, 1), (3, 1)], [(1, 2), (3, 2)]]),
       list([[(1, 1), (4, 1)], [(1, 2), (4, 2)]]),
       list([[(1, 2), (1, 2)], [(1, 3), (1, 3)]]),
       list([[(1, 2), (3, 2)], [(1, 3), (3, 3)]]),
       list([[(1, 3), (1, 3)], [(1, 4), (1, 4)]]),
       list([[(1, 3), (3, 3)], [(1, 4), (3, 4)]]),
       list([[(2, 0), (2, 0)], [(2, 1), (2, 1)]]),
       list([[(2, 0), (3, 0)], [(2, 1), (3, 1)]]),
       list([[(2, 0), (4, 0)], [(2, 1), (4, 1)]]),
       list([[(2, 1), (2, 1)], [(2, 2), (2, 2)]]),
       list([[(2, 1), (3, 1)], [(2, 2), (3, 2)]]),
       list([[(2, 1), (4, 1)], [(2, 2), (4, 2)]]),
       list([[(2, 2), (2, 2)], [(2, 3), (2, 3)]]),
       list([[(2, 2), (4, 2)], [(2, 3), (4, 3)]]),
       list([[(2, 3), (3, 3)], [(2, 4), (3, 4)]]),
       list([[(3, 0), (3, 0)], [(3, 1), (3, 1)]]),
       list([[(3, 0), (4, 0)], [(3, 1), (4, 1)]]),
       list([[(3, 1), (3, 1)], [(3, 2), (3, 2)]]),
       list([[(3, 1), (4, 1)], [(3, 2), (4, 2)]]),
       list([[(3, 2), (3, 2)], [(3, 3), (3, 3)]]),
       list([[(3, 3), (3, 3)], [(3, 4), (3, 4)]]),
       list([[(4, 0), (4, 0)], [(4, 1), (4, 1)]]),
       list([[(4, 1), (4, 1)], [(4, 2), (4, 2)]]),
       list([[(4, 2), (4, 2)], [(4, 3), (4, 3)]]),
       list([[(0, 0), (0, 0)], [(0, 1), (0, 1)], [(0, 2), (0, 2)]]),
       list([[(0, 0), (2, 0)], [(0, 1), (2, 1)], [(0, 2), (2, 2)]]),
       list([[(0, 1), (0, 1)], [(0, 2), (0, 2)], [(0, 3), (0, 3)]]),
       list([[(0, 1), (2, 1)], [(0, 2), (2, 2)], [(0, 3), (2, 3)]]),
       list([[(0, 2), (0, 2)], [(0, 3), (0, 3)], [(0, 4), (0, 4)]]),
       list([[(0, 2), (2, 2)], [(0, 3), (2, 3)], [(0, 4), (2, 4)]]),
       list([[(1, 0), (1, 0)], [(1, 1), (1, 1)], [(1, 2), (1, 2)]]),
       list([[(1, 0), (3, 0)], [(1, 1), (3, 1)], [(1, 2), (3, 2)]]),
       list([[(1, 1), (1, 1)], [(1, 2), (1, 2)], [(1, 3), (1, 3)]]),
       list([[(1, 1), (3, 1)], [(1, 2), (3, 2)], [(1, 3), (3, 3)]]),
       list([[(1, 2), (1, 2)], [(1, 3), (1, 3)], [(1, 4), (1, 4)]]),
       list([[(1, 2), (3, 2)], [(1, 3), (3, 3)], [(1, 4), (3, 4)]]),
       list([[(2, 0), (2, 0)], [(2, 1), (2, 1)], [(2, 2), (2, 2)]]),
       list([[(2, 0), (4, 0)], [(2, 1), (4, 1)], [(2, 2), (4, 2)]]),
       list([[(2, 1), (3, 1)], [(2, 2), (3, 2)], [(2, 3), (3, 3)]]),
       list([[(2, 2), (2, 2)], [(2, 3), (2, 3)], [(2, 4), (2, 4)]]),
       list([[(2, 2), (4, 2)], [(2, 3), (4, 3)], [(2, 4), (4, 4)]]),
       list([[(3, 0), (4, 0)], [(3, 1), (4, 1)], [(3, 2), (4, 2)]]),
       list([[(3, 1), (4, 1)], [(3, 2), (4, 2)], [(3, 3), (4, 3)]]),
       list([[(3, 2), (4, 2)], [(3, 3), (4, 3)], [(3, 4), (4, 4)]]),
       list([[(4, 1), (4, 1)], [(4, 2), (4, 2)], [(4, 3), (4, 3)]])],
      dtype=object)




import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
img1 = imutils.resize(img)

indices = np.where(img1!= [0])
coordinates = zip(indices[0], indices[1])

?zip



import numpy as np
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')
image = draw_haar_like_feature(np.zeros((2, 2)),
                                0, 0, 2, 2,
                                feature_coord,
                                max_n_features=1)
image

