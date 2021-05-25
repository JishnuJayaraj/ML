#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
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


import numpy as np
from GaussianMixtureModel import GaussianMixtureModel


class GMMClassifier(object):

	def __init__(self, numComponentsPerClass, maxIterations):
		self._compsPerClass = numComponentsPerClass
		self._mit = maxIterations


	def fit(self, X, y):
		self._yUnq = np.unique(y.astype(np.int))
		
		self._gml = list()
		# get index and value pair
		for i, val in enumerate(self._yUnq):
		# For the unique values of ys collect all features
			x = X[y == val]
			if i < len(self._compsPerClass): 
				val = self._compsPerClass[i]
				# Prepare the model
			gmmodel = GaussianMixtureModel(val, self._mit)			
			gmmodel.fit(x)
			self._gml.append(gmmodel)


	def predict(self, X):
		(r,c) = X.shape		
		Z = np.empty(r, np.int)
		p = np.zeros(r, np.float)
		
		# Get the index and value pair
		for i, val in enumerate(self._yUnq):
			tempP = self._gml[i].getProbabilityScores(X)
			idx = tempP > p
			p[idx] = tempP[idx]
			Z[idx] = val
		return Z


