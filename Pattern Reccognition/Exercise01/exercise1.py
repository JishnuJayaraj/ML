#%%

import scipy.misc
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

face = scipy.misc.face(gray = True)


face_gauss = scipy.ndimage.gaussian_filter(face,3)


plt.imshow(face_gauss, cmap = 'gray')

#plt.imshow(face)

# returns copy of an array collapsed into one dimension
face_flat = np.ndarray.flatten(face_gauss)

face_flat_cumsum = np.cumsum(face_flat)

face_flat_normalize = face_flat_cumsum/(np.amax(face_flat_cumsum)*1.0)


density_new = np.zeros(face.shape)


for i in range(0,400000):

    random_number = np.random.uniform()

    number = np.searchsorted(face_flat_normalize, random_number)



    position_x = number % density_new.shape[1]
    position_y = int(number / density_new.shape[1])

    density_new[position_y,position_x] = 1
    
    i = i+1


plt.imshow(density_new, cmap = 'gray')


size = (10,10)
kernel = np.ones(size)

Parzen_window = scipy.signal.convolve2d(density_new,kernel, mode = 'same', boundary = 'symm')


plt.imshow(Parzen_window, cmap = 'gray')
#scipy.sta

# Cross validation
# https://scikit-learn.org/stable/modules/cross_validation.html

#np.flattening
#np.cumsum
#normailze
#pdf
#cdf