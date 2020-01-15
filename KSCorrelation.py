#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 07:51:42 2020

@author: lorenzo
"""

from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import cluster
import scipy as sp

'''Parameters'''

alpha = 0.1
path = '/run/media/lorenzo/My Passport/General'
path = path + '/NoiseClusters.mat'

plt.close('all')
plt.clf()
plt.close()

'''Mat file to dataframe for working in python or from csv'''
df = Mat_to_dataframe(path)

'''Stadarization of the data'''
df['Mean'] = df.Bulk.apply(lambda row:np.mean(row,axis = 0))
df['Mean'] = (df['Mean'] - df.Mean.apply(lambda row:np.mean(row)))/df.Mean.apply(lambda row:np.std(row))


lenght = len(df.Mean)
'''Initialization of Matrices'''

Mean = []
for i in range(len(df.Mean)):
    Mean.append(df.Mean[i])
Correlation_ks = np.zeros((lenght,lenght))
P_Value_ks = np.zeros((lenght,lenght))

Correlation_p = np.zeros((lenght,lenght))
P_Value_p = np.zeros((lenght,lenght))


'''KS and Pearson tests'''
for i in range(lenght):
    for j in range(lenght):
        Correlation_ks[i][j],P_Value_ks[i][j] = stats.ks_2samp(df.Mean[i],df.Mean[j])
        Correlation_p[i][j],P_Value_p[i][j] = stats.pearsonr(df.Mean[i],df.Mean[j])
        
        
'''Threshold = sqrt(-0.5ln(alpha)*(m + n )/(m*n)'''
threshold = np.sqrt(-np.log(alpha)/lenght) #m = n = lenght, For KS_test
threshold2 = 0.25## For Pearson is defined as 1 - correlation


#
#sbn.clustermap(Correlation_ks,metric='euclidean',method = 'complete')
#plt.figure(1)
#linkage_ks = cluster.hierarchy.linkage(Correlation_ks,method = 'complete',metric = 'euclidean')
#plt.figure(2)
#Z_ks = cluster.hierarchy.dendrogram(linkage_ks,  color_threshold = threshold)


sbn.clustermap(Correlation_p,metric='correlation',method = 'complete')
plt.figure()
linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
#linkage_p = cluster.hierarchy.linkage(P_Value_p,method = 'complete',metric = 'euclidean')
Z_p = cluster.hierarchy.dendrogram(linkage_p,color_threshold = threshold2)


fl = cluster.hierarchy.fcluster(linkage_p,threshold2,criterion = 'distance')
#fl = cluster.hierarchy.fcluster(linkage_ks,threshold,criterion = 'distance')




def get_cluster_indexs(number,fl):
    if isinstance(number,list):
        indexes_list = []
        for i in number:
            indexes_list.append(np.where(fl == i)[0])
        return indexes_list
    else: return np.where(fl == number)[0]

def plot_branch(cluster,fl,s_plot):
    tup = get_cluster_indexs(cluster,fl)
    spikes = df.Mean[tup]
    for element in spikes:s_plot.plot(element)
    return tup

def Mat_to_dataframe(path):
    Data = loadmat(path)
    Data = Data['NoiseClusters']
    Data = Data[0]
    Column_names = Data.dtype.names
    Dataframe = {}
    for i in range(len(Column_names)):
        Dataframe[Column_names[i]] = []
    for i in range(len(Data)):
        row = Data[i]
        for j in range(len(row)):
            column = row[j]
            Dataframe[Column_names[j]].append(column)
    Data = pd.DataFrame.from_dict(Dataframe)
    return Data
fig = plt.figure(10,figsize=(10,10))
axes = fig.subplots(int(max(fl)/4),4)
axes_flt = axes.flat
for i in range(1,max(fl)):
    plot_branch(i,fl,axes_flt[i-1])
    axes_flt[i-1].title.set_text('Team {}'.format(i))
    




