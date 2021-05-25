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


import cvxopt as cv
import numpy as np


class KernelSVM(object):

	def __init__(self, C = 1.0, gamma = 0.5):
		self.width = gamma
		self.C = C
		return None


	def fit(self, X, y):
		self.X = X
		self.y = y
		X_train = self.X
		row = np.shape(X_train)[0]
		col = np.shape(X_train)[1]
		y_train = self.y
		y_train = y_train.astype(np.int)
		
		classes = list(set(y_train))
		y_train[y_train < 0 ] = -1
		y_train[y_train != -1] = +1
		k = np.zeros([col,col])
		print("Before sv gen")
		k = self.GaussianRBFKernelMatrix(X_train, X_train)
		P = cv.matrix(np.outer(y, y) * k)
		q = cv.matrix(-np.ones(row))
		G = cv.matrix(np.vstack([np.diag(-np.ones(row)),np.diag(np.ones(row))]))
		h = cv.matrix(np.hstack([np.zeros(row), self.C * np.ones(row)]))
		#A = cv.matrix(y_train.reshape(1,row).astype(np.double))
		A = cv.matrix(1.0*y_train, (1, row))
		b = cv.matrix(0.0) # argument must be a double
		qpSol = cv.solvers.qp(P, q, G, h, A, b)
		#self.lmd = np.array(qpSol['x']).flatten()
		self.lmd = np.ravel(qpSol['x'])
		print(self.lmd)
		selectInd = self.lmd > 0.001 * self.C
		#self.sm = np.array(self.lmd)[np.where(self.lmd > 0.001 * self.C)]
		self.sm = self.lmd[selectInd]
		self.sv = X[selectInd]
		self.ysv = y_train[selectInd]
		print(selectInd)
		print(self.ysv)
		#bias
		sv_bias_ind =  (self.lmd > 0.001 * self.C) & (self.lmd < self.C - 0.001*self.C)
		print(sv_bias_ind)
		y_e = np.multiply(self.lmd,y_train)*k.T
		self.bias = 0.0;
		B = self.ysv - self.predict(self.sv, mapping = False) ##y_e[sv_bias_ind].T
		self.bias = np.mean(B)
		print("Bias",self.bias)
		return None


	def GaussianRBFKernelMatrix(self, X1, X2):
		sig = self.width
		print(X1)
		print(X2)
		#k = (np.exp(-np.sum(np.power(np.subtract(X1,X2), 2),axis=0) / (2 * np.power(sig, 2))))
		k = -2*np.dot(X1, X2.T) + np.sum(np.power(X1,2),axis=1).reshape(-1,1) + np.sum(np.power(X2,2),axis=1)
		print("Before sig",k)
		k *= - sig
		print("Before K",k)
		k = np.exp(k,k)
		print("GaussianRBFKernelMatrix",k)
		return k


	def predict(self, X, mapping = True):
		z = self.GaussianRBFKernelMatrix(X, self.sv)
		K = z*self.sm * self.ysv
		K = K.sum(axis = 1) + self.bias
		if mapping:
			print("pred if")
			K[K > 0] = +1
			K[K <= 0] = -1
		print("predic",K)
		return K