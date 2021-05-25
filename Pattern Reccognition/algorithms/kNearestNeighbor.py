#	Copyright 2016 Stefan Steidl
#	Friedrich-Alexander-Universität Erlangen-Nürnberg
#	Lehrstuhl für Informatik 5 (Mustererkennung)
#	Martensstraße 3, 91058 Erlangen, GERMANY
#	stefan.steidl@fau.de


#	This file is part of the Python Classification Toolbox.
#
#	The Python Classification Toolbox is free software: 
#	you can redistribute it and/or modify it under the terms of the 
#	GNU General Public License as published by the Free Software Foundation, 
#	either version 3 of the License, or (at your option) any later version.
#
#	The Python Classification Toolbox is distributed in the hope that 
#	it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#	warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#	See the GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with the Python Classification Toolbox.  
#	If not, see <http://www.gnu.org/licenses/>.


import math
import numpy


class kNearestNeighbor(object):
	
	def __init__(self,k):
		self.__k = k

		
	def fit(self, X, y):
		# store references to the labeled training data
		self.__X = X
		self.__y = y
		self.__m = len(X)
		self.__mMax = 1e8
		
	
	def predict(self, X):
		# |x-y|^2 = (x - y)^T (x - y) = - 2 * x^T y + x^T x + y^T y 
		# runtime efficient as for-loops are avoided, but runs out of memory 
		# pretty fast for large training and test sets;
		# process only m test samples at a time
		
		m = int(self.__mMax / self.__m)
		numRuns = math.ceil(len(X) / m)
		Y = list(set(self.__y))
		Y = numpy.sort(Y).astype(numpy.int)
		z = numpy.zeros(0)
		for i in range(numRuns):
			Xs = X[i * m: (i + 1) * m]
			d1 = numpy.square(Xs).sum(axis = 1)
			d2 = numpy.square(self.__X).sum(axis = 1)
			D = numpy.dot(Xs, self.__X.T)
			D *= -2
			D += d1.reshape(-1, 1)
			D += d2
			 
			# Taking K values 
			ind = numpy.argsort(D, axis = 1)[:, 0:self.__k]   
			
			# taking total number of class
			cNum = numpy.empty((0,len(Xs)))
			eleY = self.__y[ind]
			for i in Y:
				cNum = numpy.vstack((cNum, (eleY==i).sum(axis=1)))
				
			# Taking the index of highest class number
			HighIndex = numpy.argmax(cNum, axis = 0)
			z = numpy.append(z, HighIndex)
			
		# prepare Class lebels
		lbl = numpy.zeros(len(X))
		for j in range(len(Y)):
			lbl[z == j] = Y[j]
			
		return lbl


