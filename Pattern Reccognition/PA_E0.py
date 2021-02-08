# -*- coding: utf-8 -*-

# PATTERN ANALYSIS PROGRAMMING 
# WARM-UP EXERCISES 

#%% import the useful libraries
# for image processing / pattern recognition, a useful
# collection of well-integrated packages is typically numpy (numerics),
# scipy (various mathematical operations), pandas (input/output, data tables),
# and matplotlib (visualization/plotting), see https://scipy.org
import numpy as np
import matplotlib.pyplot as plt

#%% Exercise 1
#################################################
# LISTS, TUPLES, DICTIONARIES, and NUMPY ARRAYS #
#################################################
print(5*'*', 'Exercise 1', 5*'*')

# Manually filling a list
empty_list = [] # create an empty list
list1 = [1, 2, 3]
print(list1)
print('The length is:', len(list1))
print('The first element is:', list1[0], 'and the last element is:', list1[-1])

# Manually filling a tuple
empty_tuple = () # create an empty tuple
tuple1 = (1, 2, 3, 4)
print('The length is:', len(tuple1))
print('The first element is:', tuple1[0], 'and the last element is:', tuple1[-1])

# Manually filling a dictionary
empty_dictionary = {} # create an empty dictionary
for keys in range(10):
    empty_dictionary[keys] = keys + 1
print(empty_dictionary.items()) # see the elements in the dictionary


# mnually filling a numpy array
my_np_array_1d = np.array([1,2,3])
print('The dimension of this numpy array is:', my_np_array_1d.shape)
print('The first dimension of this numpy array is:', my_np_array_1d.shape[0])
print('This numpy array is a', len(my_np_array_1d.shape), 'dimensional array')

# manually creating a 2-d numpy array
my_np_array_2d = np.array([[1,2,3],[4,5,6]]) 
print('The number of elements in this 2-d aray is:', np.size(my_np_array_2d))

# numpy random normal number, with the shape (3,2)
random_nparray = np.random.randn(3,2)

# a list is a container. Each element can be a list, tuple, dictionary, numpy array, etc.
list2 = [list1, tuple1, empty_dictionary, my_np_array_1d, my_np_array_2d]

# concatenate two lists
list_concatenated = list1 + list2

#%% Exercise 2
#################################################
########### NUMPY ARRAY MANIPULATION ############
#################################################
print('\n')
print(22 * '*')
print(5*'*', 'Exercise 2', 5*'*')

# concatenate two numpy arrays
nparray1 = np.random.randn(2,3)
nparray2 = np.random.randn(2,3)
nparrays_concatenated1 = np.concatenate((nparray1,nparray2), axis=0) # concatenate vertically
print('The shape of the vertically concatenated nparrays is:', nparrays_concatenated1.shape)
nparrays_concatenated2 = np.concatenate((nparray1,nparray2), axis=1) # concatenate vertically
print('The shape of the horizontally concatenated nparrays is:', nparrays_concatenated2.shape)

# tile (repeating an array in different axis direction)
tile_array = np.tile(nparray1, (2,3)) # repeate nparray1 2 times vertically and 3 times horizontally

#%% Exercise 3
#################################################
################### FOR LOOP ####################
#################################################
print('\n')
print(22 * '*')
print(5*'*', 'Exercise 3', 5*'*')

# normal for loop using the native python 'range' command for iteration
for i in range(5):
    print(i)
print(5*'+', '\n')

# using underline _ if we do not need a variable (here the iterator) 
list_forLoop = []
for _ in range(5):
    list_forLoop.append(np.random.randint(20, 100, (1))[0])
print(list_forLoop)
print(5*'+', '\n')

# using numpy 'arange' function instead of the native python 'range'
for _ in np.arange(5):
    print(np.random.rand(1)[0])

#%% Exercise 4
#################################################
######## INDEXING AND MAGIC INDEXING =P #########
#################################################
print('\n')
print(22 * '*')
print(5*'*', 'Exercise 4', 5*'*')

# generate a random 2-d numpy array
my_np_array_2d_2 = np.random.randint(5,30,(6,5))

# take the 2nd row and invert it
row = my_np_array_2d_2[1]
print('The last element of this vector is:', row[-1])
print('2nd row:', row)
row = row[::-1] # last element is now the first element
print('inverted 2nd row:', row)

# take the 3rd column and invert it
column = my_np_array_2d_2[:,2]
print('The 3rd column:', column)
column_inverted = column[::-1]
print('The inverted 3rd column', column_inverted)

#%% Exercise 5
#################################################
#################### IMAGES #####################
#################################################
print('\n')
print(22 * '*')
print(5*'*', 'Exercise 5', 5*'*')

from skimage import data

print('gray-scale image:', 20 * '*')

# grayscale image
checkerboard = data.checkerboard() # an image
plt.figure()
plt.imshow( checkerboard, cmap='gray' ) # show an image
plt.show( )
print( 'image data type is:', checkerboard.dtype )

print('the pixel intensity of pixel (50,100) is:', checkerboard[50,100])

checkerboard = checkerboard.astype('double') # change the type to double
checkerboard = (checkerboard - checkerboard.min()) / (checkerboard.max() - checkerboard.min()) # min-max normalization

checkerboard[:,50:100] = 0 # making a column black
checkerboard[100:150] = 1 # making a row white (other option: checkerboard[100:150,:] = 1)

plt.figure()
plt.imshow(checkerboard, cmap='gray') # show an image
plt.show()

################################################################
print('color image:', 20 * '*')

# color image
cat = data.chelsea()

plt.figure()
plt.imshow(cat) # show an image
plt.show()

print('the pixel intensity of pixel (50,100) is:', cat[50,100,:]) # RGB intensity values

cat = cat.astype('double') # change the type to double
cat = (cat - cat.min()) / (cat.max() - cat.min()) # min-max normalization

cat[:,50:100] = 0 # making a column black (other option: cat[:,50:100,:] = 0)
cat[100:150] = 1 # making a row white (other option: cat[100:150,:,:] = 1)

plt.figure()
plt.imshow(cat) # show an image
plt.show()

#%% Exercise 6
#################################################
#########  TESTING / TRAINING  SPLIT ############
#################################################
print('\n')
print(22 * '*')
print(5*'*', 'Exercise 6', 5*'*')

# a single split
from sklearn.model_selection import train_test_split
X_1 = np.random.rand(100,2)
y_1 = np.random.randint(0,3,size=(100))
print('Total samples:', y_1.size)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.4, random_state=0)
print('Train samples:', y_train.size)
print('Test samples:', y_test.size)

# spliting for k-folds cross-validation
from sklearn.model_selection import KFold
X_2 = np.random.rand(100,2)
k = 5
kf = KFold(n_splits=k)
for i, (train_index, test_index) in enumerate(kf.split(X_2)):
    X_train = X_2[train_index]
    X_test = X_2[test_index]
    print("Fold: %s; Train samples: %s; Test samples: %s" % (i, X_train.shape, X_test.shape))