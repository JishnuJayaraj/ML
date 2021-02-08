# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:34:36 2019

@author: kraus
https://vsoch.github.io/2013/the-gap-statistic/

"""
from sklearn.datasets import make_gaussian_quantiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.spatial


def data_creation():
    
    # Construct dataset
    # Gaussian 1
    X1, y1 = make_gaussian_quantiles(cov=3.,n_samples=100, n_features=2, n_classes=1)
    X1 = pd.DataFrame(X1,columns=['x','y'])
    y1 = pd.Series(y1)

    # Gaussian 2
    X2, y2 = make_gaussian_quantiles(mean=(4, 4), cov=1, n_samples=100, n_features=2, n_classes=1)
    X2 = pd.DataFrame(X2,columns=['x','y'])
    y2 = pd.Series(y2)
    
    
    X3, y3 = make_gaussian_quantiles(mean=(-6,-1),cov=3.,n_samples=100, n_features=2, n_classes=1)
    X3 = pd.DataFrame(X3,columns=['x','y'])
    y3 = pd.Series(y3)
    
    
    X4, y4 = make_gaussian_quantiles(mean = (3, -2), cov=3.,n_samples=100, n_features=2, n_classes=1)
    X4 = pd.DataFrame(X4,columns=['x','y'])
    y4 = pd.Series(y4)
    # Combine the gaussians
    X1.shape
    X2.shape
    X3.shape
    X4.shape
    
    X = pd.DataFrame(np.concatenate((X1, X2, X3, X4)))
    y = pd.Series(np.concatenate((y1, - y2 + 1, y3, y4)))
    X.shape
    
    
    plt.figure()
    plt.plot(X[0][0:100],X[1][0:100], 'ro')
    plt.plot(X[0][100:200],X[1][100:200], 'yo')
    plt.plot(X[0][200:300],X[1][200:300], 'go')
    plt.plot(X[0][300:400],X[1][300:400], 'o')



    plt.show()
    
    return X
    




def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    

gap_std = np.zeros(14)
for i in range (0,3):

    cluster = data_creation()
    cluster = np.asarray(cluster)



    k ,gap = optimalK(cluster, nrefs = 5, maxClusters = 15)


    gap = np.asarray(gap)
    
    gap_tmp = gap[:,1]
    gap_std = np.vstack((gap_std,gap_tmp))
    

gap_std = np.delete(gap_std, 0, axis = 0)
gap_std = gap_std.T




distance_matrix = scipy.spatial.distance_matrix(cluster,cluster)

within_cluster_distance_before = np.sum(distance_matrix) /2



reference = np.random.rand(100, 2)


plt.figure()

kmeans = KMeans(n_clusters=k)
a = kmeans.fit_predict(cluster)
plt.scatter(cluster[:, 0], cluster[:, 1], c=a)
plt.xlabel('k='+str(k))
plt.tight_layout()
plt.show()



c_ind0 = np.argwhere(a==0)
c_ind1 = np.argwhere(a==1)
c_ind2 = np.argwhere(a==2)
c_ind3 = np.argwhere(a==3)

cluster_distance_0 = 0
for i in c_ind0:
    for j in c_ind0:
        cluster_distance_0 += distance_matrix[i[0]][j[0]]


cluster_distance_1 = 0
for i in c_ind1:
    for j in c_ind1:
        cluster_distance_1 += distance_matrix[i[0]][j[0]]
    
    
cluster_distance_2 = 0
for i in c_ind2:
    for j in c_ind2:
        cluster_distance_2 += distance_matrix[i[0]][j[0]]
        

cluster_distance_3 = 0
for i in c_ind3:
    for j in c_ind3:
        cluster_distance_3 += distance_matrix[i[0]][j[0]]



complete_within_cluster_distance_after = cluster_distance_0/2*len(c_ind0) + cluster_distance_1/2*len(c_ind1)+ cluster_distance_2/2*len(c_ind2) +cluster_distance_3/2*len(c_ind3)

within_cluster_distance_decrease = within_cluster_distance_before - complete_within_cluster_distance_after




gap_std = np.std(gap_std,axis = 1)

plt.figure()
plt.plot(gap[:,0],gap[:,1])
plt.errorbar(gap[:,0],gap[:,1], yerr = gap_std)
plt.show()







