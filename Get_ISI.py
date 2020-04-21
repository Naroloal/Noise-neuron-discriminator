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

data = pd.read_pickle('Datos/Data_all_sessions')

experiments_path= '/run/media/lorenzo/My Passport/Data/'
experiment_list= data.PatientExperiment.unique()
experiment_data_path_list = []

for experiment in experiment_list:  
    experiment = experiment.replace('\\','/')
    experiment_data_path = experiments_path + experiment
    experiment_data_path_list.append(experiment_data_path)

ISI_df = pd.DataFrame()
for experiment in experiment_data_path_list:
    os.chdir(experiment)
    aux = experiment.split('/')[-3:]
    PatientExperiment = ','.join(aux)
    Times_files_name_list = glob.glob('times_NSX*.mat')
    df = get_isi_from_timefiles(Times_files_name_list,PatientExperiment)
    ISI_df = pd.concat([ISI_df,df],ignore_index = True)



# df = load_times_files('times_NSX2.mat')
# classes = set(df.cluster)


# plt.close('all')
# class_1 = df[df.cluster == 1]
# xa =  np.diff(class_1.time)
# hist = np.histogram(xa,bins = 100,range = {0,100})
# plt.bar(hist[1][:-1],hist[0])
# plt.legend()