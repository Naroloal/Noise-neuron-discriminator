#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:55:51 2020

@author: lorenzo
"""

import pandas as pd
from Funciones_auxiliares import load_time_files, get_isi_from_timefiles
import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
'''This script is designed to extract ISI_data from the times_files and store them in a df'''

#We need to get the path to the data, there are several experiments, so we use a previous dataframe to get the names
#of the sessions
data = pd.read_pickle('Datos/Data_all_sessions_cleaned2')
experiments_path= '/run/media/lorenzo/My Passport/Data/'
experiment_list= data.PatientExperiment.unique()
experiment_data_path_list = []

for experiment in experiment_list:  
    experiment = experiment.replace('\\','/')
    experiment_data_path = experiments_path + experiment
    experiment_data_path_list.append(experiment_data_path)

#We get the ISI and store them in a dataframe
ISI_df = pd.DataFrame()
for experiment in experiment_data_path_list:
    os.chdir(experiment)
    aux = experiment.split('/')[6:] #PatientExperiment
    PatientExperiment = '\\'.join(aux)#PatientExperiment
    Times_files_name_list = glob.glob('times_NSX*.mat')# All the times_files names in a list
    if len(Times_files_name_list) == 0: 
        print(experiment,' no tiene Times_files')
    else:
        df = get_isi_from_timefiles(Times_files_name_list,PatientExperiment) #We get the ISI data from this experiment
        ISI_df = pd.concat([ISI_df,df],ignore_index = True)# We add the df to the ISI_df



# df = load_times_files('times_NSX2.mat')
# classes = set(df.cluster)



# plt.close('all')
# class_1 = df[df.cluster == 1]
# xa =  np.diff(class_1.time)
# hist = np.histogram(xa,bins = 100,range = {0,100})
# plt.bar(hist[1][:-1],hist[0])
# plt.legend()