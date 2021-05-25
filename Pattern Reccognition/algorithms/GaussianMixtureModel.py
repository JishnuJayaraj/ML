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


import math
import numpy
import numpy.matlib

from KMeansClustering import KMeansClustering


class GaussianMixtureModel(object):
	
	def __init__(self, numComponents, maxIterations = 500):
		self._k = numComponents
		self._mIt = maxIterations
		return None


	def fit(self, X):
		self._X = X
		(r,c) = X.shape
		
		# Create prior array
		self._pr = numpy.zeros(self._k)
		#creating cov matrix
		self._cvm = numpy.zeros((self._k, c, c))
		#Creating mean matrix
		self._mu = numpy.zeros((self._k, c))
		#Create a matrix to hold old prior values for comparison
		self._cvmOld  = numpy.zeros((self._k,c,c))		
		w = numpy.zeros((self._k, r))
		# Data from KMean
		KMCls = KMeansClustering(self._k)
		idx = KMCls.fit(self._X)
		for i in range(self._k):
			xk = self._X[idx == i]
			self._pr[i] = len(xk)/r
			self._cvm[i] = numpy.cov(xk.T)
			self._mu = numpy.mean(xk, axis = 0)
			
		# counter to track EM iteration
		itr = 0
		p = numpy.zeros((self._k, r))
		while ((itr < self._mIt) and (self.covDistance(self._cvmOld, self._cvm) > 0.00001)):
			
			itr = itr+1
			# Begin Expectation steps			
			for i in range(self._k):
				p[i] = self.evaluateGaussian(self._X, self._pr[i], self._mu[i], self._cvm[i])
			w = p/numpy.sum(p, axis = 0)
			self._cvmOld = numpy.copy(self._cvm)
			# Begin Maximization step
			ws = numpy.sum(w, axis = 1)
			for i in range(self._k):
				self._pr[i] = ws[i]/r
				self._mu[i] = numpy.sum(w[i] * self._X.T) / ws[i]
				dif = numpy.subtract(self._X,self._mu[i])
				self._cvm[i] = numpy.cov((self._X-self._mu[i]).T*numpy.tile(numpy.sqrt(w[i]),(2, 1)))*r / ws[i]
		return None


	def getComponents(self, X):
		r = X.shape[0]
		Z = numpy.zeros(r, numpy.int)
		p = numpy.empty(N,numpy.float)
		p[:] = numpy.inf 
		for i in range(self._k):
			cvmInv = numpy.linalg.inv(self._cvm)
			dif = X-self.mean[i]
			# Take the dif of the log of the square of the vals
			logSqr = numpy.square(2*numpy.pi*numpy.sqrt(numpy.linalg.det(self._cvm[i])))
			pk = (numpy.log(logSqr) - numpy.log(numpy.square(self._pr[i]))) + numpy.sum(numpy.dot(numpy.dot(dif.T,cvmInv),dif),0)			
			ind = pk < p
			p[ind] = pk[ind]
			Z[ind] = i			
		return Z


	def getProbabilityScores(self, X):
		pk = numpy.zeros(X.shape[0])
		for k in range(self._k):
			pk += self.evaluateGaussian(X, self._pr[k], self._mu[k], self._cvm[k])
		return pk


	def evaluateGaussian(self, X, prior, mean, cov):
		cvmInv = numpy.linalg.inv(cov)
		dif = numpy.subtract(X,mean)
		matMult = numpy.dot(cvmInv,dif.T)*dif.T
		pk = numpy.exp(-0.5*numpy.sum(matMult,0))/ numpy.sqrt(numpy.power(2.0 * numpy.pi, X.shape[1]) * numpy.linalg.det(cov))
		pk *= prior
		return pk


	def covDistance(self, covs1, covs2):
		cvmDiff = covs1 - covs2
		cvDist = numpy.sum(numpy.abs(cvmDiff))
		return cvDist


