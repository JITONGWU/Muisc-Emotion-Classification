# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:58:20 2019

@author: hp
"""
import csv
import numpy as np
import random
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

#############read x##########

rootdir = "default_features"
file_names = []
for parents, dirnames,filenames in os.walk(rootdir):
    file_names = filenames
file_names.sort(key = lambda x: int(x[:-4]))

x = []
line = 0
for filename in file_names :
    with open(rootdir + '/' + filename) as f:
        reader = csv.reader(f,delimiter=";")
        features = []
        for row in reader:
            features.append(row)
    features = np.array(features)
    features = features[1:,1:]
    features = np.array(features).astype(np.float)
    avg_features_1 = np.mean(features[30:41],axis=0)
    x.append(avg_features_1)
    avg_features_2 = np.mean(features[42:53],axis=0)
    x.append(avg_features_2)
    avg_features_3 = np.mean(features[54:65],axis=0)
    x.append(avg_features_3)
    avg_features_4 = np.mean(features[66:77],axis=0)
    x.append(avg_features_4)
    avg_features_5 = np.mean(features[78:90],axis=0)
    x.append(avg_features_5)

    print('file '+str(line)+': read'+filename+'finish')
    line = line + 1

x = np.array(x)
print('x shape:')
print(x.shape)
##############read y333333333333
#for arousal and valence seperately
ya = []
with open('arousal_cont_average.csv') as arou:
    reader = csv.reader(arou)
    arousal = []
    for row in reader:
        arousal.append(row)
arousal = np.array(arousal)
arousal = arousal[1:,1:]
arousal = np.array(arousal).astype(np.float)
for i in range(arousal.shape[0]):
    ya.append(np.mean(arousal[i,1:12]))
    ya.append(np.mean(arousal[i,12:23]))
    ya.append(np.mean(arousal[i,24:35]))
    ya.append(np.mean(arousal[i,36:47]))
    ya.append(np.mean(arousal[i,48:61]))


yv = []
with open('valence_cont_average.csv') as val:
    reader = csv.reader(val)
    valence = []
    for row in reader:
        valence.append(row)
valence = np.array(valence)
valence = valence[1:,1:]
valence = np.array(valence).astype(np.float)
for i in range(valence.shape[0]):
    yv.append(np.mean(valence[i,1:12]))
    yv.append(np.mean(valence[i,12:23]))
    yv.append(np.mean(valence[i,24:35]))
    yv.append(np.mean(valence[i,36:47]))
    yv.append(np.mean(valence[i,48:61]))
    
y = np.vstack((ya,yv)).transpose()
print('y shape:')
print(y.shape)

#########k means clustering##########
estimator=KMeans(n_clusters=10)
res=estimator.fit_predict(y)
lable_pred=estimator.labels_
y_1d = np.zeros(y.shape[0])
for i in range(len(y)):
    y_1d[i] = int(lable_pred[i])
    

########save data#########

np.save('features.npy', x) 
np.save('cluster.npy', y_1d)
