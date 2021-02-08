# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:44:55 2019

@author: kraus
"""

import pandas as pd
import numpy as np
import math
import hmmlearn.hmm
import sklearn.preprocessing
import matplotlib.pyplot as plt



def feature_extraction(data):
    
    data = pd.read_csv(data, header = 1)
    
    data_arr = np.asarray(data)
    
    
    x = data_arr[:,0]
    y = data_arr[:,1]
    p = data_arr[:,3]
    
    timestamp = data_arr[:,2]
    timestamp_diff = np.diff(timestamp)
    timestamp_diff = timestamp_diff + 0.000001
    
    
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    
    x_diff = x_diff + 0.000001
    y_diff = y_diff + 0.000001
    
    
    x_derivate = x_diff / timestamp_diff
    y_derivate = y_diff / timestamp_diff
    
    target_angle = np.arctan(y_derivate/x_derivate)
    
    
    target_angle_diff = np.diff(target_angle)
    target_angle_diff = target_angle_diff + 0.000001
    
    
    target_angle_derivate = target_angle_diff / timestamp_diff[:-1]
    
    path_vel = np.sqrt(x_derivate*x_derivate + y_derivate*y_derivate)
    path_vel_diff = np.diff(path_vel)
    path_vel_derivation = path_vel_diff / timestamp_diff[-1]
    
    
    target_angle_derivate = np.abs(target_angle_derivate)
    
    log_curv_rad = np.log10(path_vel[:-1]/target_angle_derivate)
    
    c_n = path_vel[:-1] * target_angle_derivate
    
    tot_acc_mag = np.sqrt(path_vel_derivation*path_vel_derivation + c_n*c_n)
    
    x = sklearn.preprocessing.scale(x)
    x = x[:-2]
    y = sklearn.preprocessing.scale(y)
    y= y[:-2]
    p = sklearn.preprocessing.scale(y)
    target_angle = sklearn.preprocessing.scale(target_angle)
    target_angle= target_angle[:-1]
    path_vel = sklearn.preprocessing.scale(path_vel)
    path_vel= path_vel[:-1]
    log_curv_rad = sklearn.preprocessing.scale(log_curv_rad)
    tot_acc_mag = sklearn.preprocessing.scale(tot_acc_mag)
    
    length = len(tot_acc_mag)
    
    return x , y, p, target_angle, path_vel, log_curv_rad, tot_acc_mag, length


h_values=[1,2,4]
m_values = [1, 2, 4, 8, 16]
tab_orig = np.zeros((3,5))
tab_imit =np.zeros((3,5))
for h in range(len(h_values)):
    for m in range(len(m_values)):
        nStates = h_values[h]
        nMix = m_values[m]

        startprob = np.zeros(nStates)
        startprob[0] = 1
        transmat = np.triu(np.ones((nStates, nStates)))
        transmat = transmat / np.expand_dims(np.sum(transmat, axis=1), axis=1)

        
        
        
        matrix = np.empty(7)
        length_list = []
        for i in range(1,26):

            if i<= 9:
                data = 'pa_mobisig/user_1/original_0' + str(i) + '.csv'
            else:
                data = 'pa_mobisig/user_1/original_' + str(i) + '.csv'
            x, y, p, target_angle, path_vel, log_curv_rad, tot_acc_mag, length = feature_extraction(data)
            
            b = np.array([x, y, p, target_angle, path_vel, log_curv_rad, tot_acc_mag])
            
            
            length_list.append(length)
            
            matrix = np.c_[matrix,b]


        matrix = np.delete(matrix, np.s_[0:1], axis = 1)
        length_list = np.asarray(length_list)
        length_list_cumsum = np.cumsum(length_list)
        test_matrix = np.empty(7)
        test_length_list = []
        for i in range(1,40):

            if i<= 9:
                data = 'pa_mobisig/user_1/imitated_0' + str(i) + '.csv'
            elif i >25:
                data = 'pa_mobisig/user_1/original_' + str(i) + '.csv'
            x, y, p, target_angle, path_vel, log_curv_rad, tot_acc_mag, length = feature_extraction(data)
            
            b = np.array([x, y, p, target_angle, path_vel, log_curv_rad, tot_acc_mag])
            
            
            test_length_list.append(length)
            
            test_matrix = np.c_[matrix,b]

        test_matrix = np.delete(test_matrix, np.s_[0:1], axis = 1)
        test_length_list = np.asarray(test_length_list)
        test_length_list_cumsum = np.cumsum(test_length_list)

        model = hmmlearn.hmm.GMMHMM(n_components=nStates, covariance_type="diag", n_mix = nMix )

        #model = hmmlearn.hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="cm", params="cm")
        model.startprob_ = startprob

        model.transmat_ = transmat
        #hmmlearn.hmm.GaussianHMM(n_components = 3,covariance_type='diag')

        model.fit(matrix.T, length_list)

        model.transmat_ = transmat

        thresh = 0
        for i in range(0,24):
            d = matrix[0:7,length_list_cumsum[i]: length_list_cumsum[i+1]]
            thresh = np.min([thresh, model.score(d.T)])
            
        original_average = 0
        imit_average = 0    
        test_probs = np.zeros(20)
        testDecisions = np.zeros(20)
        for i in range(0,19):
            d = test_matrix[0:7,test_length_list_cumsum[i]: test_length_list_cumsum[i+1]]
            s = model.score(d.T)
            test_probs[i] = model.score(d.T)
            if s >= thresh:
                testDecisions[i] = 1
            if i<10:
                imit_average+=s
            else:
                original_average+=s
                
            
        model.transmat_ = transmat
            
            
        train_probs = np.zeros(25)
        for i in range(0,24):
            d = matrix[0:7,length_list_cumsum[i]: length_list_cumsum[i+1]]
            s = model.score(d.T)
            train_probs[i] = model.score(d.T)

            



        fig = plt.figure()

        x_train = np.arange(0, 24)
        plt.plot(x_train, train_probs[0:24], 'bo')
        x_test = np.arange(24, 33)
        plt.plot(x_test, test_probs[10:19], 'go')
        x_imit = np.arange(33,42)
        plt.plot(x_imit, test_probs[0:9], 'ro')
        plt.xlabel('0-24: training data; 25-44: original test data; 45-64: imitated data')
        plt.ylabel('score output')

        plt.show()



        original_average = int(original_average / (10))
        imit_average = int(imit_average / (10))
        tab_imit[h, m] = imit_average
        tab_orig[h, m] = original_average


        print("Original Data")
        print(str(tab_orig))
        print(" ")
        print(" ")
        print(" ")
        print("Imitated Data")
        print(str(tab_imit))
    print("Original Data")
    print(str(tab_orig))
    print(" ")
    print(" ")
    print(" ")
    print("Imitated Data")
    print(str(tab_imit))
