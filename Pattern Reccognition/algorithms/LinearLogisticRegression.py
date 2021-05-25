#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-UniversitÃ¤t Erlangen-NÃ¼rnberg
#    Lehrstuhl fÃ¼r Informatik 5 (Mustererkennung)
#    MartensstraÃŸe 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software: 
#    you can redistribute it and/or modify it under the terms of the 
#    GNU General Public License as published by the Free Software Foundation, 
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that 
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.  
#    If not, see <http://www.gnu.org/licenses/>.


import numpy
import math
import numpy.matlib


class LinearLogisticRegression(object):

    def __init__(self, learningRate = 0.5, maxIterations = 100):
        self.lr = learningRate
        self.mi = maxIterations
        return None


    def fit(self, X, y):
        self.__y = y
        self.__X = X
        self.__nX = len(X[0])+1
        self.__nY = len(y)

		# Theta random
        Theta = numpy.array([1, 2, 3]) #Choosing random values as 1,2,3
        Theta.shape = (3,1) # Changing to row vector

        iter = 0
        # Newton-Raphson iteration
        while (iter < self.mi):
            gd = numpy.zeros((self.__nX,1)) # Gradient
            ll = 0 # log
            for i in range(self.__nY):
                x = numpy.append(X[i,:],1)
                l = self.__y[i] # Labels
                sig = self.gFunc(x, Theta)

                if (sig > 0 and sig < 1):
                    ll = ll + (l*numpy.log(sig) + (1-l)*numpy.log(1-sig))

                # Gradient computation
                f = l - sig
                gd = numpy.add(gd,f*numpy.array(x)[:, numpy.newaxis])

            Theta = numpy.add(Theta,self.lr*gd)
            iter += 1 
        self.theta = Theta
        return None


    def gFunc(self, x, theta):
        z = numpy.dot(x,theta)
        if(z < 0):
            return 1 - (1 / (1 + math.exp(z)))
        return 1 / (1 + math.exp(-z))


    def predict(self, X):
        nl = X.shape[0]
        X1 = numpy.ones((nl, 1))
        Xn = numpy.hstack((X,X1))
        label = numpy.dot(Xn, self.theta)
        label[numpy.where(label>0)] = 1
        label[numpy.where(label<0)] = 0
        return label


