#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:36:09 2020

@author: lorenzo
"""

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
from Funciones_auxiliares import Mat_to_dataframe,see_teams,get_cluster_indexs,plot_branch,plot_teams
from scipy import cluster
import scipy as sp
import math
from scipy.spatial.distance import squareform
import itertools
import chart_studio.plotly as plotly

plt.close('all')
plt.clf()
plt.close()
##############################
'''Mat file to dataframe for working in python'''
NOISE = pd.read_pickle('NOISE')
NEURON = pd.read_pickle('NEURON')
NOISE_MAYBE = pd.read_pickle('Noise_Maybe')


df = pd.concat([NOISE,NEURON],ignore_index = True,sort = False)
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
        Correlation_p[i,j],P_Value_p[i,j] = stats.pearsonr(df.Mean[i],df.Mean[j])
######################################################
threshold2 = 0.1## For Pearson is defined as 1 - correlation
######################################################
'''Plot matrix correlation clustered'''
#Clustermap = sbn.clustermap(Mean,metric='correlation',method = 'complete')
#plt.figure(2)
linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
#linkage_p = cluster.hierarchy.linkage(P_Value_p,method = 'complete',metric = 'euclidean')
Z_p = cluster.hierarchy.dendrogram(linkage_p,color_threshold = threshold2,leaf_font_size = 8)
fl = cluster.hierarchy.fcluster(linkage_p,threshold2,criterion = 'distance')
 ########################################################
## =============================================================================
df['Team'] = 0
for i in range(1,max(fl)+1):
    tup = get_cluster_indexs(i,fl)
    df['Team'][tup] = i
## =============================================================================
#'''Plot one spikeshape per team'''
#fig = plt.figure(4)
#axes = fig.subplots(int(math.ceil(len(Important)/4)),4)
#for i in range(len(Important)):
#     axes.flat[i].plot(Important.Mean.iloc[i])
#     axes.flat[i].title.set_text('Team {}'.format(Important.Team.iloc[i]))
#'''Now we introduce the good spikeshapes'''
#df_good = df[df.bNoise == 0]
#gb = pd.concat([df_good,Important])
#Important = pd.concat([Important,df[df.bNoise == 0]])
## =============================================================================
#'''Plot one spikeshape per team'''
#fig = plt.figure(5)
#axes = fig.subplots(int(math.ceil(len(Important)/4)),4)
#for i in range(len(Important)):
#     axes.flat[i].plot(Important.Mean.iloc[i])
#     axes.flat[i].title.set_text('Team {}'.format(Important.Team.iloc[i]))
#'''Now we introduce the good spikeshapes'''
#df_good = df[df.bNoise == 0]
#gb = pd.concat([df_good,Important])
## =============================================================================
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

#########################################################################        
'''HOW IS THE RELATION BETWEEN THRESHOLD AND NUM OF MIXED Teams??'''
threshold = np.linspace(0,1,1000)
num_teams_correlated_per_threshold = []
for t in threshold:
    linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
    fl = cluster.hierarchy.fcluster(linkage_p,t,criterion = 'distance')
    Mix_teams = []
    for team in np.arange(1,max(fl)+1):
        spikes_shapes = get_cluster_indexs(team,fl)
        aux = list(df.bNoise[spikes_shapes])
        aux2 = []
        for element in aux:
            aux2.append(element)
            if  (aux2.count(1) !=0) and (aux2.count(0)!=0):
                Mix_teams.append(team)
    num_teams_correlated_per_threshold.append(len(Mix_teams))      
fig = plt.figure()
plt.scatter(threshold,num_teams_correlated_per_threshold,c = 'r',marker = '+',label = 'Mixed Teams')
plt.xlabel = ('Threshold')
plt.ylabel('Num_mixed_teams')







#############################################################################
#############################################################################
#'''KS test. Doesn't work well.'''
#lenght = len(NOISE)
#threshold_ks =  1.073*np.sqrt(128/64**2)
#Correlation_ks = np.zeros((lenght + 1,lenght+1))
#P_Value_ks = np.zeros((lenght+1,lenght+1))
#for i in range(lenght):
#    for j in range(lenght):
#        Correlation_ks[i,j],P_Value_ks[i,j] = stats.ks_2samp(NOISE.Mean.iloc[i],NOISE.Mean.iloc[j])
#
#for i in range(len(NEURON)):
#    r = NOISE.append(NEURON.iloc[i],ignore_index = True,sort = False)
#    c = pd.DataFrame(np.array(r.Mean.tolist()))
#    Correlation_ks_lc= [stats.ks_2samp(r.Mean.iloc[lenght],r.Mean.iloc[k])[0] for k in range(lenght + 1)]
#    Correlation_ks[lenght] = Correlation_ks_lc
#    Correlation_ks[:,181] = Correlation_ks_lc
#    condensed = squareform(Correlation_ks)
#    linkage_ks = cluster.hierarchy.linkage(condensed,method = 'complete')
#    fl = cluster.hierarchy.fcluster(linkage_p,threshold_ks,criterion = 'distance')
#    Mix_teams = []
#    for team in np.arange(1,max(fl)+1):
#        spikes_shapes = get_cluster_indexs(team,fl)
#        aux = list(df.bNoise[spikes_shapes])
#        aux2 = []
#        for element in aux:
#            aux2.append(element)
#            if  (aux2.count(1) !=0) and (aux2.count(0)!=0):
#                Mix_teams.append(team)
#                print('spike {} mixed'. format(i))
#
#    

    
