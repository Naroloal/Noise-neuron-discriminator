#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 06:28:32 2020

@author: lorenzo
"""

import pandas as pd
import Funciones_auxiliares as fa
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# In[]
NOISE = pd.read_pickle('FINAL_NOISE')

# In[]

plots_per_fig = 9
Num_fig = len(NOISE)//plots_per_fig + 1

figures = [plt.figure() for i in range(Num_fig)]
axes = [fig.subplots(3,3).flatten() for fig in figures]
# In[]

Bulk_dataframe = NOISE.Bulk

for i in range(len(Bulk_dataframe)):
    fig_to_plot = i//plots_per_fig
    ax_to_plot = i%plots_per_fig
    fa.plot_Bulk(Bulk_dataframe.iloc[i],axes[fig_to_plot][ax_to_plot],label = i)
    plt.legend()
