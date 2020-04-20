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
from Funciones_auxiliares import Mat_to_dataframe

# a = pd.read_pickle('Data_all_sessions')
a = Mat_to_dataframe('/run/media/lorenzo/My Passport/General/Noise_Clusters/NoiseClusters2.mat')

names = {}

a.PatientExperiment = a.PatientExperiment.apply(lambda row:row[0])
a.Cluster= a.Cluster.apply(lambda row:row[0][0])
a.Channel= a.Channel.apply(lambda row:row[0][0])
a.bUnSure= a.bUnSure.apply(lambda row:row[0][0])
a.bNoise= a.bNoise.apply(lambda row:row[0][0])  


with open('names.txt','r') as f:
    for line in f:
        line = line.split(',')
        names[line[0]] = line[1].strip()

def extract_info(string):
    a = string[4:8]
    b = string[-2:]
    if b[0].isdigit():
        info = a + b
    else: info = a+ b[1]
    return info

# a['Data'] = a.apply(lambda row: names[row.PatientExperiment] +'_'+str(row.Channel)+'_'+str(row.Cluster),axis = 1)
a['Data'] = a.apply(lambda row: extract_info((row.PatientExperiment)) +'_'+str(row.Channel)+'_'+str(row.Cluster),axis = 1)

a['Mean_nn'] = a.Bulk.apply(lambda row:np.mean(row,axis = 0))

plt.close('all')

Noise_Sure = a[(a.bNoise == 1)&(a.bUnSure == 0)]
Noise_UnSure = a[(a.bNoise == 1)&(a.bUnSure == 1)]

Neuron_Sure = a[(a.bNoise == 0)&(a.bUnSure == 0)]
Neuron_UnSure = a[(a.bNoise == 0)&(a.bUnSure == 1)]

# len_NoS = len(Noise_sure)
# len_NoU = len(Noise_unsure)
# len_NeS = len(Neuron_sure)
#len_NeU = len(Neuron_unsure)


plots_per_fig = 25

def plots_stuff(df,plot_cluster = False,save_name = ''):
    num_figs = len(df)//25+1
    Figures = [plt.figure(i,figsize = (20,10)) for i in range(num_figs)]
    axes = []
    ind_fig = 0
    print('Se realizaran '+str(num_figs) + 'figuras  con '+ str(plots_per_fig) + ' subfiguras cada una... \n')
    print('Figuras finalizadas:')
    for figure in Figures:
        axes.append(figure.subplots(5,5).flat)
        figure.tight_layout()
        figure.subplots_adjust(hspace=.3, wspace=.3)
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
        if Fig_to_plot == ind_fig  + 1:
            Figures[ind_fig].savefig(save_name + str(ind_fig))
            plt.close(str(ind_fig))
            ind_fig +=1
            print('\n Figura '+str(ind_fig) +' guardada')
        print(i)
    Figures[-1].savefig(save_name + str(ind_fig + 1))
    plt.close(str(ind_fig+1))

plots_stuff(Noise_Sure,True,'Figuras/Noise_Sure2/')
plt.close('all')
plots_stuff(Neuron_UnSure,True,'Figuras/Neuron_UnSure2/')
plt.close('all')
plots_stuff(Neuron_Sure,True,'Figuras/Neuron_Sure2/')
plots_stuff(Noise_UnSure,True,'Figuras/Noise_UnSure2/')
plt.close('all')



#######################################################
# "Indices sacados de Dataframes temporales NOS,NEU,NES,NOU. Obtenidos de mirar los plots usando Plots_stuff"
# Noise_sure_to_maybe = [80,81,85,100,101]
# Noise_unsure_to_maybe = [0,1,2,7,10,23,25,26,27,28,29,31,35,36,37,38,44,47,61,65,79,82,83]
# Neuron_unsure_to_maybe = [19]
# Noise_Sure_eliminate= [89]
# Noise_Unsure_eliminate = [47,69,70,84,86,89]

# #############################################################################################3

# NOSTM = Noise_sure.iloc[Noise_sure_to_maybe]
# NOUTM = Noise_unsure.iloc[Noise_unsure_to_maybe]
# NEUTM = Neuron_unsure.iloc[Neuron_unsure_to_maybe]
# Noise_Maybe = pd.concat([NOSTM,NOUTM,NEUTM],ignore_index=True,sort = False) 

# Noise_sure.reset_index(inplace = True)
# Noise_unsure.reset_index(inplace = True)
# Neuron_unsure.reset_index(inplace = True)

# Noise_sure.drop(labels = np.concatenate((Noise_sure_to_maybe,Noise_Sure_eliminate)),inplace = True)
# Noise_unsure.drop(labels = np.concatenate((Noise_unsure_to_maybe,Noise_Unsure_eliminate)),inplace = True)
# Neuron_unsure.drop(labels = Neuron_unsure_to_maybe,inplace = True)

###############################################################################################


# NOISE = pd.concat([Noise_sure,Noise_unsure],ignore_index = True,sort = False)
# NOISE.drop(columns = ['bUnSure','index'],inplace = True)
# NEURON = pd.concat([Neuron_sure,Neuron_unsure],ignore_index = True,sort = False)
# NEURON.drop(columns = ['bUnSure','index'],inplace = True)

# pd.to_pickle(NOISE,'NOISE')
# pd.to_pickle(NEURON,'NEURON')
# pd.to_pickle(Noise_Maybe,'Noise_Maybe')






        

        
    
