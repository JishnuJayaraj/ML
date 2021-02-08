

import scipy.misc
import scipy.ndimage
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

face = scipy.misc.face(gray = True)


face_gauss = scipy.ndimage.gaussian_filter(face,3)


plt.imshow(face_gauss, cmap = 'gray')

#plt.imshow(face)

face_flat = np.ndarray.flatten(face_gauss)

face_flat_cumsum = np.cumsum(face_flat)

face_flat_normalize = face_flat_cumsum/(np.amax(face_flat_cumsum)*1.0)


density_new = np.zeros(face.shape)


for i in range(0,200000):

    random_number = np.random.uniform()

    number = np.searchsorted(face_flat_normalize, random_number)



    position_x = number % density_new.shape[1]
    position_y = number / density_new.shape[1]

    density_new[position_y,position_x] = 1
    
    i = i+1

plt.figure()
plt.imshow(density_new, cmap = 'gray')


size = (10,10)
kernel = np.ones(size)

Parzen_window = scipy.signal.convolve2d(density_new,kernel, mode = 'same', boundary = 'symm')

plt.figure()
plt.imshow(Parzen_window, cmap = 'gray')



X_2 = density_new
k = 5
theta10 = 0
theta5 = 0
theta3 = 0
kf = KFold(n_splits=k)
for i, (train_index, test_index) in enumerate(kf.split(X_2)):
    X_train = X_2[train_index]
    X_test = X_2[test_index]
    
    
    Parzen_window10 = scipy.signal.convolve2d(X_train ,np.ones((10,10)), mode = 'same', boundary = 'symm')
    theta10 +=  np.sum(np.ma.log(Parzen_window10[:,test_index]))

    Parzen_window5 = scipy.signal.convolve2d(X_train ,np.ones((15,15)), mode = 'same', boundary = 'symm')
    theta5 +=  np.sum(np.ma.log(Parzen_window5[:,test_index]))
    
    Parzen_window3 = scipy.signal.convolve2d(X_train ,np.ones((3,3)), mode = 'same', boundary = 'symm')
    theta3 +=  np.sum(np.ma.log(Parzen_window3[:,test_index]))

theta10/=k
theta5/=k
theta3/=k

maximum = np.argmax([theta10,theta5,theta3])

#scipy.sta

#theta10 = np.argmax(theta10)
#theta5 = np.argmax(theta5)
#theta3 = np.argmax(theta3)

#np.flattening
#np.cumsum
#normailze
#pdf
#cdf