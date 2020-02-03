#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:31:51 2020

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
matplotlib.use('Qt5agg')
plt.ion()
'''Noise database finished, now we want to compare how it works with new noise'''

# In[]
path_new_noise = '/run/media/lorenzo/My Passport/General/NoiseClusters.mat'
New_noise = Mat_to_dataframe(path_new_noise)
New_noise.drop(columns = ['bNoise','bUnSure'],inplace = True)
# In[]
NOISE = pd.read_pickle('FINAL_NOISE')
NOISE = NOISE[['PatientExperiment','Channel','Cluster','Bulk','Mean']]
Noise_lenght = len(NOISE)
# In[]
# Preanalize the new_noise
New_noise['Mean'] = New_noise.Bulk.apply(lambda row: np.mean(row,axis = 0))
New_noise['Mean'] = New_noise.Mean.apply(lambda row: (row - np.mean(row))/np.std(row))
New_noise_lenght = len(New_noise)
# In[]
df = pd.concat((NOISE,New_noise),ignore_index = True,sort = False)
# In[]
Means = pd.DataFrame(np.array(df.Mean.tolist()).transpose())
Correlation = Means.corr()
# In[]
# We just care for the correlation between old and new noise
Correlation = Correlation.iloc[:,:Noise_lenght]
Correlation = Correlation.iloc[Noise_lenght:]
# In[]
# Noise founded and not founded with a given Threshold, along with the index (in Correlation df) of the maximum correlation
Threshold = 0.90
noise_founded = set(np.where(Correlation.values>Threshold)[0])
noise_not_founded = set(np.arange(New_noise_lenght)) - noise_founded
max_corr_id_noise_not_founded = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_not_founded]
max_corr_id_noise_founded = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_founded]

# In[]
def plot_max_correlated(list1,list2,df,Noise_lenght = Noise_lenght):
    fig = plt.figure()
    num_columns = mt.ceil(np.sqrt(len(list1)))
    axes = fig.subplots(num_columns,num_columns)
    axes_flt = axes.flatten()
    j = 0
    max_corr = [np.max(Correlation.iloc[i].tolist()) for i in list1]
    for i,idx in zip(list1,list2):
        C = max_corr[j]
        axes_flt[j].plot(df.Mean.iloc[i + Noise_lenght])
        axes_flt[j].plot(df.Mean.iloc[idx])
        axes_flt[j].set_title('Correlation = {}'.format(round(C,3)))
        j +=1
# In[]
    plot_max_correlated(noise_founded,max_corr_id_noise_founded,df)
    plot_max_correlated(noise_not_founded,max_corr_id_noise_not_founded,df)
# In[]
def add_new_means_to_noise_df(Not_founded,max_corr_id_not_founded,NOISE,New_meanspikes):
    Noise_lenght = len(New_noise)
    df = pd.concat((New_noise,NOISE),ignore_index = True,sort = False)
    for i,idx in zip(Not_founded,max_corr_id_not_founded):
        plt.close(fig = 4)
        fig = plt.figure(4)
        plt.plot(df.Mean.iloc[i + Noise_lenght],label = 'Not_founded')
        plt.plot(df.Mean.iloc[idx],label = 'Noise')
        plt.legend()
        plt.show()
        plt.pause(1)
        print('Add Not founded to Noise df? y/n')
        a = input()
        if a == 'y':
            NOISE.append(df.iloc[i+Noise_lenght])
            print('Not_founded spike added')
        elif a == 'n':
            print('Not added')
        else: raise Exception('not valid response')
    return NOISE

NOISE = add_new_means_to_noise_df(noise_not_founded,max_corr_id_noise_not_founded,NOISE,New_noise)
