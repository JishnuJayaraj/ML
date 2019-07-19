'''
pluralsight.comnetwork seperate data from 2 blobs dataset

'''


from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# remove warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Helper Functions

# plot data on figure
def plot_data(pl, X, y):
    # plot class where y == 0
    pl.plot(X[y==0,0], X[y==0,1], 'ob', alpha=0.5)
    # plot class where y == 1
    pl.plot(X[y==1,0], X[y==1,1], 'xr', alpha=0.5)

    pl.legend(['0', '1'])
    return pl

# function that draws decision boundary
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make predicion with the model and reshape the op so contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12,8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alph=0.2)
    # plt the moons of data
    plot_data(plt, X,y)

    return plt


# Generate some data blobs. data will be either 0 or 1 (when no: center=2)
# X is a [no of samples, 2] sized array. X[sample] contains its x,y position ofthe sample in the space
# y is a [no of samples] sized array. y[sample] contains the class index (0 or 1)

X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

pl = plot_data(plt, X, y)
pl.show()