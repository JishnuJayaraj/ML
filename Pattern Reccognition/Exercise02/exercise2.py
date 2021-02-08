# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:12:05 2019

@author: kraus
"""
from sklearn.datasets import make_regression


import scipy.misc
import scipy.ndimage
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import sklearn.ensemble
import sklearn.datasets

#%%
face = scipy.misc.face(gray = True)


face_gauss = scipy.ndimage.gaussian_filter(face,3)


plt.imshow(face_gauss, cmap = 'gray')


density_new = np.zeros(face.shape)


face_flat = np.ndarray.flatten(face_gauss)

face_flat_cumsum = np.cumsum(face_flat)

face_flat_normalize = face_flat_cumsum/(np.amax(face_flat_cumsum)*1.0)
x_coord = []
y_coord = []

for i in range(0,200000):

    random_number = np.random.uniform()

    number = np.searchsorted(face_flat_normalize, random_number)



    position_x = number % density_new.shape[1]
    position_y = number / density_new.shape[1]

    density_new[position_y,position_x] = 1
    
    i = i+1
    
    x_coord.append(position_x)
    y_coord.append(position_y)
    
    


plt.imshow(density_new, cmap = 'gray')


density_background = np.zeros(face.shape)

random_x = np.random.uniform(0,1023,200000)
random_x = np.int_(random_x)

random_y = np.random.uniform(0,767,200000)
random_y = np.int_(random_y)


density_background[random_y,random_x] = 1



x_coord = np.asarray(x_coord)
y_coord = np.asarray(y_coord)

x_coord = np.append(x_coord,random_x)
y_coord = np.append(y_coord,random_y)


label = np.zeros(400000)
label[0:200000] = 1


X = [x_coord,y_coord]
X = np.asarray(X).T


#regr = sklearn.ensemble.RandomForestRegressor()
#regr = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, max_depth = 10)
regr = sklearn.ensemble.ExtraTreesRegressor(n_estimators = 5, max_depth = 10)



regr.fit(X,label)



X_predict = np.zeros([786432,2])

X_predict= np.int_(X_predict)
counter = 0
for y in range (0,767):
    for x in range(0,1023):
        X_predict[counter,0]=x
        X_predict[counter,1]=y
        counter = counter +1


regression_racoon = regr.predict(X_predict)


predict_array = np.zeros(face.shape)



for i in range(0, len(X_predict)):

    predict_array[X_predict[i,1], X_predict[i,0]] = regression_racoon[i]


plt.imshow(predict_array,cmap = 'gray')
plt.show()
#%%

#%%
tmp = sklearn.datasets.make_moons()[0]

tmp[:,1] +=1
tmp[:,1] *=408 -1

tmp[:,0] +=1
tmp[:,0] *=256 -1

tmp=np.int_(tmp)


moon = np.zeros(face.shape)
moon[tmp[:,0],tmp[:,1]]=1

moon_gauss = scipy.ndimage.gaussian_filter(moon,3)


plt.imshow(moon_gauss, cmap = 'gray')


density_new = np.zeros(moon.shape)


moon_flat = np.ndarray.flatten(moon_gauss)

moon_flat_cumsum = np.cumsum(moon_flat)

moon_flat_normalize = moon_flat_cumsum/(np.amax(moon_flat_cumsum)*1.0)
x_coord = []
y_coord = []

for i in range(0,1000):

    random_number = np.random.uniform()

    number = np.searchsorted(moon_flat_normalize, random_number)



    position_x = number % density_new.shape[1]
    position_y = number / density_new.shape[1]

    density_new[position_y,position_x] = 1
    
    i = i+1
    
    x_coord.append(position_x)
    y_coord.append(position_y)



density_background = np.zeros(face.shape)



random_x = np.random.uniform(0,1023,1000)
random_x = np.int_(random_x)

random_y = np.random.uniform(0,767,1000)
random_y = np.int_(random_y)

density_background[random_y,random_x] = 0.5


x_coord = np.asarray(x_coord)
y_coord = np.asarray(y_coord)

x_coord = np.append(x_coord,random_x)
y_coord = np.append(y_coord,random_y)


label = np.zeros(2000)
label[0:1000] = 1


X = [x_coord,y_coord]
X = np.asarray(X).T


regr = sklearn.ensemble.RandomForestRegressor()



regr.fit(X,label)


X_predict = np.zeros([786432,2])

X_predict= np.int_(X_predict)
counter = 0
for y in range (0,767):
    for x in range(0,1023):
        X_predict[counter,0]=x
        X_predict[counter,1]=y
        counter = counter +1


regression_moon = regr.predict(X_predict)


predict_array = np.zeros(face.shape)



for i in range(0, len(X_predict)):

    predict_array[X_predict[i,1], X_predict[i,0]] = regression_moon[i]


plt.imshow(predict_array,cmap = 'gray')
plt.show()

#%%







