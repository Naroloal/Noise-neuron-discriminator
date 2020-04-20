#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:00:15 2020

@author: lorenzo
"""

# In[]
import pandas as pd
from Funciones_auxiliares import Mat_to_dataframe
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
import time as t
import os
import glob
matplotlib.use('Qt5agg')
plt.ion()
'''Noise database finished, now we want to try with ALL the clusters from an experiment. I created this script thinking in a 
database where is not know what is noise and what isn't. Also considering that there are multiunints in tehe database   '''
plt.close('all')
# In[]
# New Data loading, I don't know which is noise and which is not.
NoiseClusters = 'NoiseClusters4.mat'
path_new_data = '/run/media/lorenzo/My Passport/General/' + NoiseClusters
New_Data = Mat_to_dataframe(path_new_data)
New_Data.drop(columns = ['bNoise','bUnSure'],inplace = True)
New_Data['Mean'] = New_Data.Bulk.apply(lambda row: np.mean(row,axis = 0))
New_Data['Mean'] = New_Data.Mean.apply(lambda row: (row - np.mean(row))/np.std(row))

# In[]
#Old database loading, also the new neurons are incorporated to the NEURON database
NOISE = pd.read_pickle('FINAL_NOISE2')
NOISE = NOISE[['PatientExperiment','Channel','Cluster','Bulk','Mean']]
Noise_lenght = len(NOISE)
df = pd.concat((NOISE,New_Data),ignore_index = True,sort = False)

# In[]
Means = pd.DataFrame(np.array(df.Mean.tolist()).transpose())
Correlation = Means.corr()
# In[]
# We just care for the correlation between old and new data
Correlation = Correlation.iloc[:,:Noise_lenght]
Correlation = Correlation.iloc[Noise_lenght:]

# In[]
#Now let's see what we found. Not_noise is the mean spikeshapes that weren't found by the correlatoin process. It does not mean
#is actually a good response. The same with noise founded
Threshold = 0.95
noise_founded = set(np.where(Correlation.values>Threshold)[0])
Not_noise = set(np.arange(len(New_Data))) - noise_founded
max_corr_id_not_noise = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_not_founded]
max_corr_id_noise_founded = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_founded]
# In[]
plot_max_correlated(noise_founded,max_corr_id_noise_founded,df)

# In[]
def plot_max_correlated(list1,list2,df,Noise_lenght = Noise_lenght):
    '''Function to see the max correlation between the means spikes shapes in df
    The indexes lists should be in list1 and list2. Where the indexes of list1 represent the elements of df
    minus Noise_lenght.
    
    So for example list1 = [0,1,5], list2 = [1,14,13]. then element Noise_length + 0 in df has it's maximum correlation
    with element 1  in df. While Noise_lenght +5 with element 13 in df'''
    Num_figs = len(list1)//25+1
    Num_plots_per_figure = 25
    Columns =int(np.sqrt(Num_plots_per_figure))
    fig_list = [plt.figure() for _ in range(Num_figs)]
    axes_list = [fig.subplots(Columns,Columns) for fig in fig_list]
    axes_flt = [axes.flatten() for axes in axes_list]
    j = 0
    max_corr = [np.max(Correlation.iloc[i].tolist()) for i in list1]
    for i,idx in zip(list1,list2):
        fig_to_plot = j//Num_plots_per_figure
        ax_to_plot = j%Num_plots_per_figure
        C = max_corr[j]
        axes_flt[fig_to_plot][ax_to_plot].plot(df.Mean.iloc[i + Noise_lenght])
        axes_flt[fig_to_plot][ax_to_plot].plot(df.Mean.iloc[idx])
        axes_flt[fig_to_plot][ax_to_plot].set_title('Correlation = {}'.format(round(C,3)))
        j +=1
    plt.show()
    plt.pause(1)
