#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:29:36 2020

@author: lorenzo
"""

import pandas as pd
import matplotlib.pyplot as plt
from Funciones_auxiliares import Mat_to_dataframe,see_teams,get_cluster_indexs,plot_branch,plot_Bulk
import numpy as np
from scipy import cluster
import scipy as sp
import math

plt.close('all')

data = pd.read_pickle(('Datos/Data_all_sessions_cleaned'))
data['Mean'] = (data['Mean'] - data.Mean.apply(lambda row:np.mean(row)))/data.Mean.apply(lambda row:np.std(row))

'''Selection of the data, import patientes 30,31,32'''
df = data[data.PatientExperiment.str.contains('HEC032|HEC031|HEC030')]
df.reset_index(inplace = True)

'''Mean in Matrix'''
Mean = pd.DataFrame(np.array(df.Mean.to_list())).values

'''Threshold'''
threshold2 = 0.1## For Pearson is defined as 1 - correlation

'''Plot matrix correlation clustered'''
plt.figure(2)
linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
fl = cluster.hierarchy.fcluster(linkage_p,threshold2,criterion = 'distance')

'''Has every Team only good/bad spikeshapes???''' ###Depends on the threshold??
Mix_teams = []
print('There are mix teams??')
for team in np.arange(1,max(fl)+1):
    spikes_shapes = get_cluster_indexs(team,fl)
    aux = list(df.bNoise[spikes_shapes])
    aux2 = []
    for element in aux:
          aux2.append(element)
    if  (aux2.count(1) !=0) and (aux2.count(0)!=0):
        print('Yes, team {} is a mix team'.format(team))
        Mix_teams.append(team)
        
'''Plot mixed teams'''
if not len(Mix_teams) ==0:
    fig = plt.figure(4)
    axes = fig.subplots(int(math.ceil(len(Mix_teams)/2)),2)
    for i in range(len(Mix_teams)):
        plot_branch(Mix_teams[i],fl,axes.flat[i],df,label=True)
        axes.flat[i].title.set_text('Team {}'.format(Mix_teams[i]))
        
for team in Mix_teams:
    fig = plt.figure()
    indexes = get_cluster_indexs(team,fl)
    axes = fig.subplots(int(math.ceil(len(indexes)/2)),2).flat
    to_plot = df.loc[indexes,['Bulk','bNoise','Mean']]
    for i,(bulk,bnoise,mean) in enumerate(zip(to_plot.Bulk,to_plot.bNoise,to_plot.Mean)):
        plot_Bulk(bulk,axes[i])
        plt.plot(mean,label = bnoise)
        plt.legend()
    


