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
import matplotlib.cm as cm
from Funciones_auxiliares import Mat_to_dataframe,see_teams,get_cluster_indexs,plot_branch
from scipy import cluster
import scipy as sp
import math

#################################
'''Parameters'''
alpha = 0.1
path = '/run/media/lorenzo/My Passport/General/NoiseClusters.mat'
path2 = '/run/media/lorenzo/My Passport/General/NoiseClusters2.mat'
#########################################
plt.close('all')
plt.clf()
plt.close()
##############################
'''Mat file to dataframe for working in python'''
df = Mat_to_dataframe(path)
df2 = Mat_to_dataframe(path2)
df = pd.concat([df,df2],ignore_index = True,sort = False)
df.bNoise = df.bNoise.apply(lambda row: row[0][0])
df.bUnSure = df.bUnSure.apply(lambda row: row[0][0])
#########################################################################
'''Stadarization of the data'''
df['Mean'] = df.Bulk.apply(lambda row:np.mean(row,axis = 0))
df['Mean'] = (df['Mean'] - df.Mean.apply(lambda row:np.mean(row)))/df.Mean.apply(lambda row:np.std(row))

'''Just Bad/good spikes (Comment both if you want to work with the entire dataframe)'''
#df = df[df.bNoise == 1]
#df = df.reset_index()
#df.drop(columns = 'index',inplace = True)
##df = df[df.bNoise == 0]
lenght = len(df.Mean)

#######################################################################
'''Initialization of Matrices'''
Mean = []
for i in range(len(df.Mean)):
    Mean.append(df.Mean[i])

Correlation_p = np.zeros((lenght,lenght))
P_Value_p = np.zeros((lenght,lenght))

########################################################################
'''Pearson tests'''
for i in range(lenght):
    for j in range(lenght):
        Correlation_p[i][j],P_Value_p[i][j] = stats.pearsonr(df.Mean[i],df.Mean[j])
######################################################
'''Threshold = sqrt(-0.5ln(alpha)*(m + n )/(m*n)'''
threshold2 = 0.1## For Pearson is defined as 1 - correlation
######################################################
'''Plot matrix correlation clustered'''
#Clustermap = sbn.clustermap(Correlation_p,metric='correlation',method = 'complete')
plt.figure(2)
linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
#linkage_p = cluster.hierarchy.linkage(P_Value_p,method = 'complete',metric = 'euclidean')
#   Z_p = cluster.hierarchy.dendrogram(linkage_p,color_threshold = threshold2)
fl = cluster.hierarchy.fcluster(linkage_p,threshold2,criterion = 'distance')
#=============================================================================
 ########################################################
'''Plot Teams'''
fig = plt.figure(3,figsize=(10,10))
axes = fig.subplots(int(max(fl)/4)+1,4)
axes_flt = axes.flat
for i in range(1,max(fl)+1):
    plot_branch(i,fl,axes_flt[i-1],df)
    axes_flt[i-1].title.set_text('Team {}'.format(i))
########################################################
#=============================================================================
##=============================================================================
# ########################################################
#'''Plot Teams'''
#fig = plt.figure(3,figsize=(10,10))
#axes = fig.subplots(int(max(fl)/4)+1,4)
#axes_flt = axes.flat
#for i in range(1,max(fl)+1):
#    plot_branch(i,fl,axes_flt[i-1],df)
#    axes_flt[i-1].title.set_text('Team {}'.format(i))
#########################################################
## =============================================================================
#'''We just need one member for each Team.'''
#s = []
#df['Team'] = 0
#for i in range(1,max(fl)+1):
#    tup = get_cluster_indexs(i,fl)
#    s.append(tup[0])
#    df['Team'][tup] = i
#Important = df.iloc[s]
#
#Important = pd.concat([Important,df[df.bNoise == 0]])
## =============================================================================
#'''Plot one spikeshape per team'''
#fig = plt.figure(4)
#axes = fig.subplots(int(math.ceil(len(Important)/4)),4)
#for i in range(len(Important)):
#     axes.flat[i].plot(Important.Mean.iloc[i])
#     axes.flat[i].title.set_text('Team {}'.format(Important.Team.iloc[i]))
#
#'''Now we introduce the good spikeshapes'''
#df_good = df[df.bNoise == 0]
#gb = pd.concat([df_good,Important])
#
#Important = pd.concat([Important,df[df.bNoise == 0]])
## =============================================================================
#'''Plot one spikeshape per team'''
#fig = plt.figure(5)
#axes = fig.subplots(int(math.ceil(len(Important)/4)),4)
#for i in range(len(Important)):
#     axes.flat[i].plot(Important.Mean.iloc[i])
#     axes.flat[i].title.set_text('Team {}'.format(Important.Team.iloc[i]))
#
'''Now we introduce the good spikeshapes'''
#df_good = df[df.bNoise == 0]
#gb = pd.concat([df_good,Important])
##
#
## =============================================================================
#'''Has every Team only good/bad spikeshapes???''' ###Depends on the threshold??
#Mix_teams = []
#print('There are mix teams??')
#for team in np.arange(1,max(fl)+1):
#    spikes_shapes = get_cluster_indexs(team,fl)
#    aux = list(df.bNoise[spikes_shapes])
#    aux2 = []
#    for element in aux:
#          aux2.append(element)
#    if  (aux2.count(1) !=0) and (aux2.count(0)!=0):
#        print('Yes, team {} is a mix team'.format(team))
#        Mix_teams.append(team)
#
#'''Plot mixed teams'''
#if not len(Mix_teams) ==0:
#    fig = plt.figure(4)
#    axes = fig.subplots(int(math.ceil(len(Mix_teams)/2)),2)
#    for i in range(len(Mix_teams)):
#        plot_branch(Mix_teams[i],fl,axes.flat[i],df,label=True)
#        axes.flat[i].title.set_text('Team {}'.format(Mix_teams[i]))
#        
#'''Plot mixed teams'''
#if not len(Mix_teams) ==0:
#    fig = plt.figure(4)
#    axes = fig.subplots(int(math.ceil(len(Mix_teams)/2)),2)
#    for i in range(len(Mix_teams)):
#        plot_branch(Mix_teams[i],fl,axes.flat[i],df,label=True)
#        axes.flat[i].title.set_text('Team {}'.format(Mix_teams[i]))

#
#
#
#
#'''Dasddas'''
#
#Save = df[(df.Team ==24 )| (df.Team ==32)| (df.Team == 41) ]
#cmap = plt.get_cmap('gnuplot')
#colors = [cmap(i) for i in np.linspace(0, 1, 5)]
#i = 0
#for Team in Save.Team.unique():
#    fig = figure(i,figsize = (6,9.8))
#    axes = fig.subplots(2,1)
#    D = Save[Save.Team == Team]
#    j = len(D)
#    lent = len(D.Bulk.iloc[0][0])
#    for s in range(j):
#        for element in D.Bulk.iloc[s]:
#            axes.flat[0].plot(np.arange(lent),element,c = colors[s])
#        axes.flat[1].plot(np.arange(lent),D.Mean.iloc[s],c = colors[s],label = '{},{}'.format(D.bNoise.iloc[s],D.bUnSure.iloc[s]))
#        axes.flat[1].legend()
#        axes.flat[1].title.set_text('Team {}'.format(Mix_teams[i]))
#    i+=1
##
