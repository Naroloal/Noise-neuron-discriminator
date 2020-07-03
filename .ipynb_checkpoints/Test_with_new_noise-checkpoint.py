#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:31:51 2020

@author: lorenzo
"""


# In[]
import pandas as pd
from Funciones_auxiliares import Mat_to_dataframe,plot_Bulk
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
import time as t
import os
import glob
matplotlib.use('Qt5agg')
plt.ion()
'''Noise database finished, now we want to compare how it works with new noise'''
plt.close('all')

NoiseClusters = 'NoiseClusters5.mat'
path_new_data = '/run/media/lorenzo/My Passport/General/' + NoiseClusters
#path_new_data = '/run/media/lorenzo/BACKUP_PHD_BRN/Handsome_Lorenzo/' + NoiseClusters
New_Data = Mat_to_dataframe(path_new_data)
New_noise = New_Data[New_Data.bNoise == 1]
New_neurons = New_Data[New_Data.bNoise == 0]
try:
    New_noise.drop(columns = ['bNoise','bUnSure'],inplace = True)
    New_neurons.drop(columns = ['bNoise','bUnSure'],inplace = True)
except:
    New_noise.drop(columns = ['bNoise'],inplace = True)
    New_neurons.drop(columns = ['bNoise'],inplace = True)


# In[]
# Preanalize the new_noise
New_noise['Mean'] = New_noise.Bulk.apply(lambda row: np.mean(row,axis = 0))
New_noise['Mean'] = New_noise.Mean.apply(lambda row: (row - np.mean(row))/np.std(row))
New_noise_lenght = len(New_noise)

New_neurons['Mean'] = New_neurons.Bulk.apply(lambda row: np.mean(row,axis = 0))
New_neurons['Mean'] = New_neurons.Mean.apply(lambda row: (row - np.mean(row))/np.std(row))
# In[]
#Old database loading, also the new neurons are incorporated to the NEURON database
NOISE = pd.read_pickle('FINAL_NOISE5')
NOISE = NOISE[['PatientExperiment','Channel','Cluster','Bulk','Mean']]
NEURON = pd.read_pickle('NEURON')
#just in case
NEURON['Mean'] = NEURON.Bulk.apply(lambda row: np.mean(row,axis = 0))
NEURON['Mean'] = NEURON.Mean.apply(lambda row: (row - np.mean(row))/np.std(row))
NEURON = pd.concat((NEURON,New_neurons),ignore_index = True, sort = False)
Noise_lenght = len(NOISE)

# In[]
df = pd.concat((NOISE,New_noise),ignore_index = True,sort = False)
# In[]
#Correlation of the noise
Means = pd.DataFrame(np.array(df.Mean.tolist()).transpose())
Correlation = Means.corr()
# In[]
# We just care for the correlation between old and new noise
Correlation = Correlation.iloc[:,:Noise_lenght]
Correlation = Correlation.iloc[Noise_lenght:]
# In[]
# Noise founded and not founded with a given Threshold, along with the index (in Correlation df) of the maximum correlation
Threshold = 0.95
noise_founded = set(np.where(Correlation.values>Threshold)[0])
noise_not_founded = set(np.arange(New_noise_lenght)) - noise_founded
max_corr_id_noise_not_founded = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_not_founded]
max_corr_id_noise_founded = [np.argmax(Correlation.iloc[i].tolist()) for i in noise_founded]

print('{}% of noise was founded'.format(len(noise_founded)*100/(len(noise_founded) + len(noise_not_founded))))

# In[]
#Plots of the noise founded and not founded
plot_max_correlated(noise_founded,max_corr_id_noise_founded,df)
plot_max_correlated(noise_not_founded,max_corr_id_noise_not_founded,df)
# In[]
#Decide if incorporate noise not founded in the NOISE database

NOISE = add_new_means_to_noise_df(noise_not_founded,max_corr_id_noise_not_founded,NOISE,New_noise,ask = True)

# In[]
#We want to check if the new noise is highly correlationated with the neurons database
Noise_lenght = len(NOISE)
df = pd.concat((NOISE,NEURON),ignore_index = True,sort = False)
Means = pd.DataFrame(np.array(df.Mean.tolist()).transpose())

Corr = Means.corr()
Corr = Corr.values

#Gb stands for good and bad mean spikeshapes
Gb = Corr[Noise_lenght:,:Noise_lenght]
G_Correlated,B_Correlated = np.where(Gb > Threshold)

# In[]
if len(G_Correlated) == 0: print('It works!')
else:
    neurons_correlated_list = np.unique(G_Correlated)
    noise_index_list = [np.where(G_Correlated == neuron) for neuron in neurons_correlated_list]
    noise_correlated_list = [B_Correlated[noise_index] for noise_index in noise_index_list]
    to_plot = [np.append(neurons_correlated_list[i],noise_correlated_list[i]) for i in range(len(neurons_correlated_list))]
#    for element in to_plot:
#        plot_clusters(element,df)
    for element in to_plot:
        neuron,noise_list = element[0],element[1:]
        for noise in noise_list:
            try:
                plt.close('all')
                compare_and_drop(neuron,noise,df,Corr,NOISE,Noise_lenght)
            except:
                if noise not in df.index:print('element already erased')               
                
    # In[]
def add_new_means_to_noise_df(Not_founded,max_corr_id_not_founded,NOISE,New_meanspikes,ask = True):
    '''Function to add the new noises to the noise database. '''
    for i,idx in zip(Not_founded,max_corr_id_not_founded):
        if ask:
            C = Correlation.iloc[i,idx]
            plt.close(fig = 4)
            fig = plt.figure(4)
            plt.plot(New_noise.Mean.iloc[i],label = 'Not_founded')
            plt.plot(NOISE.Mean.iloc[idx],label = 'Noise')
            plt.title(C)
            plt.legend()
            plt.show()
            plt.pause(1)
            print('Add Not founded to Noise df? y/n')
            a = input()
            if a == 'y':
                NOISE = NOISE.append(New_noise.iloc[i],ignore_index = True)
                print('Not_founded spike added')
            elif a == 'n':
                print('Not added')
            else: raise Exception('not valid response')
        else : NOISE = NOISE.append(New_noise.iloc[i],ignore_index = True)
    return NOISE

def plot_max_correlated(list1,list2,df,Noise_lenght = Noise_lenght):
    '''Function to see the max correlation between the means spikes shapes in df
    The indexes lists should be in list1 and list2. Where the indexes of list1 represent the elements of df
    minus Noise_lenght.
    
    So for example list1 = [0,1,5], list2 = [1,14,13]. then element Noise_length + 0 in df has it's maximum correlation
    with element 1  in df. While Noise_lenght +5 with element 13 in df'''
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
    plt.show()
    plt.pause(1)

def plot_clusters(list_index_to_plot,df,Noise_lenght = Noise_lenght):
    fig = plt.figure()
    axes = fig.subplots(len(list_index_to_plot),1).flatten()
    for index,ax in zip(list_index_to_plot,axes):
        C = Corr[list_index_to_plot[0]+Noise_lenght,index]
        if list_index_to_plot[0] == index:
            color = 'b'
            Bulk = df.Bulk.iloc[index + Noise_lenght]
        else: 
            Bulk = df.Bulk.iloc[index]
            color = 'r'
        for spike in Bulk:
            ax.plot(spike,linewidth = 0.1,c = color)
        ax.set_title(C)
        
def compare_plots(neuron_id,noise_id,df,Corr,Noise_lenght = Noise_lenght):
    fig = plt.figure()
    ax = fig.subplots(2,1)
    Bulk1 = df.Bulk.loc[neuron_id + Noise_lenght]
    Bulk2 = df.Bulk.loc[noise_id]
    plot_Bulk(Bulk1,ax[0])
    plot_Bulk(Bulk2,ax[1])
    plt.show()
    plt.pause(2)
    print('Corr = {}'.format(Corr[neuron_id + Noise_lenght,noise_id]))
        
    
def plot_means(df,id1,id2,Noise_lenght = Noise_lenght):
    fig = plt.figure()
    ax = fig.subplots(2,1)
    C = np.dot(df.Mean.loc[id1 + Noise_lenght],df.Mean.iloc[id2])
    ax[0].plot(df.Mean.loc[id1 + Noise_lenght],c = 'b')
    ax[1].plot(df.Mean.loc[id2],c = 'r')
    plt.title(C/64)
    
def drop_noise(df,idx):
    df = df.drop(idx,inplace = True)
    
def compare_and_drop(neuron_id,noise_id,df,Corr,NOISE,Noise_lenght = Noise_lenght):
    compare_plots(neuron_id,noise_id,df,Corr,Noise_lenght = Noise_lenght)
    plot_means(df,neuron_id,noise_id,Noise_lenght)
    plt.show()
    plt.pause(2)
    s = 1
    while (s!='y' and s != 'n'):
        print('erase noise? y/n \n')
        s = input()
        if s == 'y':
            drop_noise(NOISE,noise_id)
        else: continue

