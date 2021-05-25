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
import scipy.optimize._minimize as optimize


class LinearRegression(object):
    
    def __init__(self, lossFunction = 'l2', lossFunctionParam = 0.001, classification = False):
        self.__initialized = True #set to true when done
        self.lf = lossFunction
        self.lfp = lossFunctionParam
        return None


    def fit(self, X, y):
        # For loss function = l2
        print("Yugal")
        if self.lf == 'l2':
            Xnew = numpy.vstack((X,numpy.ones(X.shape[0]))).T
            self.params = numpy.linalg.pinv(Xnew).dot(y)
            return None
		# For loss function = huber	
        a = self.lfp
        p1 = numpy.random.uniform(0.001,0.0008)
        p2 = numpy.random.uniform(0.001,0.0008)
        p = [p1,p2]
        self.params = p
        self.params = optimize.minimize(lambda x: self.huber_objfunc(X, y, x, a), self.params, method = 'BFGS',
                                          jac = lambda x: self.huber_objfunc_derivative(X, y, x, a)).x
        return None


    def huber_objfunc(self, X, y, params, a):
        r = y - numpy.dot(params[0],X)- params[1]
        q = numpy.sum(self.huber(r,a))
        return q


    def huber_objfunc_derivative(self, X, y, params, a):
        r = y - numpy.dot(params[0],X)- params[1]
        grad1 = numpy.sum(numpy.multiply(-X, self.huber_derivative(r, a)))
        grad2 = numpy.sum(-self.huber_derivative(r, a))
        hGrad = numpy.array([grad1,grad2])
        return hGrad


    def huber(self, r, a):
        hub = numpy.zeros(numpy.array(r).size)
        hub[abs(r) < a] = numpy.square(r[abs(r) <= a])
        hub[abs(r) >= a] = a * (2 * abs(r[abs(r) >= a]) - a)
        return hub
		
    def huber_derivative(self, r, a):
        hub_grad = numpy.zeros(numpy.array(r).size)
        hub_grad[abs(r) < a] = 2 * r[abs(r) < a]
        hub_grad[abs(r) >= a] = 2 * a * numpy.sign(r[abs(r) >= a])
        return hub_grad


    def paint(self, qp, featurespace):
        if self.__initialized:
            x_min, y_min, x_max, y_max = featurespace.coordinateSystem.getLimits()
            y1 = self.params[0] * x_min + self.params[1]
            x1, y1 = featurespace.coordinateSystem.world2screen(x_min, y1)
            y2 = self.params[0] * x_max + self.params[1]
            x2, y2 = featurespace.coordinateSystem.world2screen(x_max, y2)
            qp.drawLine(x1, y1, x2, y2)


    def predict(self, X):
        return None

