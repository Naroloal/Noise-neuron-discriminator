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
        subplot.plot(df.Mean_nn.iloc[i],'k',linewidth = 1,label = 'i = {}'.format(i))
        subplot.legend()
        

        
    
