#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:20:28 2020

@author: lorenzo
"""

"""There are several NoiseCluster*.mat in the HardDrive, the idea of this script is put all together into a single dataframe"""

import pandas as pd
import Funciones_auxiliares as fa
import glob
import numpy as np

#Path of the matfiles
Path = '/run/media/lorenzo/My Passport/General/'
List_of_matfiles = glob.glob(Path + 'NoiseClusters*.mat')

#for every path we create a dataframe, containing the data and we stored it in a dictionary
Dataframes = {}
i = 1
for matfile_path in List_of_matfiles:
    Dataframes[i] = fa.Mat_to_dataframe(matfile_path)
    i += 1
    
data = pd.DataFrame()


def convert_column(row):
    try:
        return int(row[0][0])
    except:return np.nan


    
for dataframe in Dataframes.values():
    data = pd.concat([dataframe,data],ignore_index = True,sort = False)

data.bNoise = data.bNoise.apply(convert_column)
data.bUnSure = data.bUnSure.apply(convert_column)
data.Channel = data.Channel.apply(convert_column)
data.Cluster = data.Cluster.apply(convert_column)

data.PatientExperiment = data.PatientExperiment.apply(lambda row:row[0])


data.to_pickle('Data_all_sessions')