import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
#import plotly.plotly as py
#import plotly.graph_objs as go


# import a .csv, compute the feature vector and normalize the data
def getData(file):
    x = []
    y = []
    p = []
    velX = []
    velY = []
    accX = []
    accY = []

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
            p.append(float(row['pressure']))
            velX.append(float(row['velocityx']))
            velY.append(float(row['velocityy']))
            accX.append(float(row['accelx']))
            accY.append(float(row['accely']))

    x = np.asarray(x)
    y = np.asarray(y)
    p = np.asarray(p)
    accX = np.asarray(accX)
    accY = np.asarray(accY)
    velX = np.asarray(velX)
    velY = np.asarray(velY)

    # compute 7 features from paper
    phi = np.arctan(velX / (velY + 1e-15))
    v = np.sqrt(velX ** 2 + velY ** 2)
    gradPhi = np.abs(np.gradient(phi)) + 1e-15
    absV = np.abs(v) + 1e-15
    rho = np.log(absV / gradPhi)
    a = np.sqrt(accX ** 2 + accY ** 2)

    # concatenate to feature vector
    feat = np.array([x, y, p, phi, v, rho, a])

    # compute gradient and concatenate
    # gradFeat = np.gradient(feat, axis=1)
    # featureVec = np.concatenate((feat, gradFeat), axis=0)
    featureVec = feat

    # force 0 mean and unit variance
    featureVec = featureVec - np.expand_dims(np.mean(featureVec, axis=1), axis=1)
    # featureVec = featureVec/np.expand_dims(np.sqrt(np.var(featureVec, axis=1)), axis=1)
    return np.transpose(featureVec)




def main():
    h_values=[1,2,4,8,16]
    m_values = [1, 2, 4, 8, 16, 32, 64, 128]
    tab_orig = np.zeros((6,8))
    tab_imit =np.zeros((6,8))
    for h in range(len(h_values)):
        for m in range(len(m_values)):
            success=0
            # where are the data stored?
            folder =  '/proj/ciptmp/ul62ilop/ass4/pa_mobisig'

            # list of folder paths for possible combinations of users and signatures

            users = ['user_'+ str(i) for i in range(10)]
            #print(users)

            files_imit = ['imitated_' + str(i).zfill(2) + '.csv' for i in range(1,21)]
            #print(files_imit)

            files_orig = ['original_' + str(i).zfill(2) + '.csv' for i in range(1,46)]
            #print(files_orig)
            nTrain = 25
            nTest_orig = 20
            nTest_imit = 20
            nTest = nTest_orig + nTest_imit

            nStates = h_values[h]
            nMix = m_values[m]

            startprob = np.zeros(nStates)
            startprob[0] = 1
            transmat = np.triu(np.ones((nStates, nStates)))
            transmat = transmat / np.expand_dims(np.sum(transmat, axis=1), axis=1)

            trainProbs = np.zeros((10, nTrain))

            # store log-likelihood from score function
            testProbs = np.zeros((10, nTest))

            # store hard decision: correct signature or not
            testDecisions = np.zeros((10, nTest))
            original_average = 0
            imit_average = 0
            for u in range(10):

                filesTrain = [os.path.join(folder, users[u], files_orig[o]) for o in range(nTrain)]
                filesTest_orig = [os.path.join(folder, users[u], files_orig[o]) for o in range(nTrain, nTrain + nTest_orig)]
                filesTest_imit = [os.path.join(folder, users[u], files_imit[o]) for o in range(nTest_imit)]
                filesTest = np.concatenate((filesTest_orig, filesTest_imit), axis=0)

                # initialize HMM
                #model = hmm.GaussianHMM(n_components=nStates)
                model = hmm.GMMHMM(n_components=nStates, covariance_type="diag", n_mix=4 )#mit spherical funktioniert es mit diag nicht
                model.startprob_ = startprob
                model.transmat_ = transmat

                # read the train data
                trainData = []
                lensTrain = np.zeros((nTrain))
                for i in range(nTrain):
                    d = getData(filesTrain[i])
                    lensTrain[i] = np.shape(d)[0]
                    trainData.append(d)
                trainData = np.vstack(trainData)

                # fit the model
                model.fit(trainData, lengths=lensTrain.astype(int))
                model.transmat_ = transmat
                # get a threshold for hard decision: lowest score of train set
                thresh = 0
                for i in range(nTrain):
                     d = getData(filesTrain[i])
                     thresh = np.min([thresh, model.score(d)])

                # testing
                for i in range(nTest):
                    d = getData(filesTest[i])
                    s = model.score(d)
                    testProbs[u, i] = model.score(d)
                    if s >= thresh:
                        testDecisions[u, i] = 1
                    if i < nTest_orig:
                        original_average+=s
                    if i >= nTest_orig:
                        imit_average += s
                model.transmat_ = transmat
                ###Training Data


                #print('User '+str(u)+', H = ' +str(h) + ', M = '+str(m) + ', original average: ' +str(original_average))
                #print('User ' + str(u) + ', H = ' + str(h) + ', M = ' + str(m) + ', imitated average: ' + str(
                #    imit_average))
                #if(original_average > imit_average):
                 #   print('-----SUCCESS-----')
                  #  success+=1
                #else:
                 #   print('NO SUCCESS :(')


                for i in range(nTrain):
                    d = getData(filesTrain[i])
                    s = model.score(d)
                    trainProbs[u, i] = model.score(d)

                if(h==8 and m==4):
                    print('writer ' + str(u))
                    fig, axs = plt.subplots(2, 1)
                    axs[0].plot([nTest_orig - 0.5, nTest_orig - 0.5], [np.min(testProbs[u, :]), np.max(testProbs[u, :])], 'k')
                    axs[0].plot(testProbs[u, :])
                    axs[0].plot([0, nTest - 1], [thresh, thresh], 'r')
                    axs[0].set_xlabel('0->19: original writer; 20-39: imitated')
                    axs[0].set_ylabel('score output')
                    axs[0].set_title('log-likelihood and decision threshold writer ' + str(u))
                    #plt.show()
                    #plt.figure()
                    x_train = np.arange(0, 25)
                    axs[1].plot(x_train, trainProbs[u, :], 'bo')
                    x_test = np.arange(25, 45)
                    axs[1].plot(x_test, testProbs[u, 0:20], 'go')
                    x_imit = np.arange(45,65)
                    axs[1].plot(x_imit, testProbs[u, 20:40], 'ro')
                    axs[1].set_xlabel('0-24: training data; 25-44: original test data; 45-64: imitated data')
                    axs[1].set_ylabel('score output')
                    axs[1].set_title('matching score per signature of writer ' + str(u))
                    plt.show()
                    #left side should be above threshold (original writer), right side should be below (imitated)
            #print('Number of success for H = ' +str(h) + ', M = '+str(m) + ': '+str(success))
            original_average = int(original_average / (nTest_orig*10))
            imit_average = int(imit_average / (nTest_imit*10))
            tab_imit[h, m] = imit_average
            tab_orig[h, m] = original_average

            if(h==8 and m==4):
                # some more plots

                plt.figure(figsize=(8, 2))
                plt.imshow(testProbs)
                plt.title('HMM score output (log-likelihood)')
                plt.colorbar()
                plt.xlabel('0->19: original; 20-39: imitated')
                plt.ylabel('user')
                plt.show()

                plt.figure(figsize=(8, 2))
                plt.imshow(testDecisions)
                plt.title('decision: it was the original writer')
                plt.colorbar()
                plt.xlabel('0->19: original; 20-39: imitated')
                plt.ylabel('writer')
                plt.show()

                trueVals = np.concatenate((np.ones((10, nTest_orig)), np.zeros((10, nTest_imit))), axis=1)
                trueDecs = (trueVals == testDecisions)
                plt.figure(figsize=(8, 2))
                plt.imshow(trueDecs + 0)
                plt.title('correct decisions')
                plt.colorbar()
                plt.xlabel('0->19: original; 20-39: imitated')
                plt.ylabel('writer')
                plt.show()

                plt.figure()
                plt.stem(np.sum(trueDecs, axis=1) / nTest)
                plt.plot([0, 9], [0.5, 0.5], 'r--')
                plt.title('Recognition rate for individual writers')
                plt.xlabel('writer')
                plt.ylabel('Recognition rate')
                plt.show()

                recognitionRate = np.sum(trueDecs) / (nTest * 10)
                print('Combined recognition rate: ' + str(recognitionRate * 100) + '%')
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
    #trace_orig = go.Table(header=dict(values=['M=1', 'M=2','M=4']),cells=dict(values=tab_orig))
    #data_orig = [trace_orig] #, 'M=8','M=16', 'M=32','M=64', 'M=128']
    #trace_imit = go.Table(header=dict(values=['M=1', 'M=2', 'M=4']),     cells=dict(values=tab_imit))
    #data_imit = [trace_imit] #, 'M=8', 'M=16', 'M=32', 'M=64', 'M=128']
    #py.iplot(data_orig, filename='Original Data')
    #py.iplot(data_imit, filename='Imitated Data')
if __name__ == "__main__":
    main()
