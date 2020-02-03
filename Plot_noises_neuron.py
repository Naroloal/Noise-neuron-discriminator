#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:29:47 2020

@author: lorenzo
"""


import pandas as pd
import matplotlib.pyplot as plt
import math as mt
import numpy as np

a = pd.read_pickle('Spikes')

a['Data'] = a.apply(lambda row: extract_info(row['PatientExperiment'][0])+'_'+str(row.Channel[0][0])+'_'+str(row.Cluster[0][0]),axis = 1)
a['Mean_nn'] = a.Bulk.apply(lambda row:np.mean(row,axis = 0))

plt.close('all')

Noise_sure = a[(a.bNoise == 1)&(a.bUnSure == 0)]
Noise_unsure = a[(a.bNoise == 1)&(a.bUnSure == 1)]

Neuron_sure = a[(a.bNoise == 0)&(a.bUnSure == 0)]
Neuron_unsure = a[(a.bNoise == 0)&(a.bUnSure == 1)]

len_NoS = len(Noise_sure)
len_NoU = len(Noise_unsure)
len_NeS = len(Neuron_sure)
len_NeU = len(Neuron_unsure)

plots_per_fig = 25


def plots_stuff(df,plot_cluster = False):
    num_figs = len(df)//25+1
    Figures = [plt.figure(i) for i in range(num_figs)]
    axes = []
    for figure in Figures:
        axes.append(figure.subplots(5,5).flat)

    for i in range(len(df)):
        Fig_to_plot = i//plots_per_fig
        fig = axes[Fig_to_plot]
        ax_to_plot = i%plots_per_fig
        subplot = fig[ax_to_plot]
        if plot_cluster:
            for j in range(len(df.Bulk.iloc[i])):
                subplot.plot(df.Bulk.iloc[i][j],'b',linewidth = 0.1)
        subplot.plot(df.Mean_nn.iloc[i],'k',linewidth = 1,label = '{}'.format(df.Data.iloc[i]))
        subplot.legend()
        
def extract_info(string):
    a = string[4:8]
    b = string[-2:]
    if b[0].isdigit():
        info = a + b
    else: info = a+ b[1]
    return info
#######################################################
"Indices sacados de Dataframes temporales NOS,NEU,NES,NOU. Obtenidos de mirar los plots usando Plots_stuff"
Noise_sure_to_maybe = [80,81,85,100,101]
Noise_unsure_to_maybe = [0,1,2,7,10,23,25,26,27,28,29,31,35,36,37,38,44,47,61,65,79,82,83]
Neuron_unsure_to_maybe = [19]
Noise_Sure_eliminate= [89]
Noise_Unsure_eliminate = [47,69,70,84,86,89]

#############################################################################################3

NOSTM = Noise_sure.iloc[Noise_sure_to_maybe]
NOUTM = Noise_unsure.iloc[Noise_unsure_to_maybe]
NEUTM = Neuron_unsure.iloc[Neuron_unsure_to_maybe]
Noise_Maybe = pd.concat([NOSTM,NOUTM,NEUTM],ignore_index=True,sort = False) 

Noise_sure.reset_index(inplace = True)
Noise_unsure.reset_index(inplace = True)
Neuron_unsure.reset_index(inplace = True)

Noise_sure.drop(labels = np.concatenate((Noise_sure_to_maybe,Noise_Sure_eliminate)),inplace = True)
Noise_unsure.drop(labels = np.concatenate((Noise_unsure_to_maybe,Noise_Unsure_eliminate)),inplace = True)
Neuron_unsure.drop(labels = Neuron_unsure_to_maybe,inplace = True)

###############################################################################################


NOISE = pd.concat([Noise_sure,Noise_unsure],ignore_index = True,sort = False)
NOISE.drop(columns = ['bUnSure','index'],inplace = True)
NEURON = pd.concat([Neuron_sure,Neuron_unsure],ignore_index = True,sort = False)
NEURON.drop(columns = ['bUnSure','index'],inplace = True)

pd.to_pickle(NOISE,'NOISE')
pd.to_pickle(NEURON,'NEURON')
pd.to_pickle(Noise_Maybe,'Noise_Maybe')






        

        
    
