#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:55:51 2020

@author: lorenzo
"""

import pandas as pd
from Funciones_auxiliares import load_times_files
import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = pd.read_pickle('Datos/Data_all_sessions')

experiments_path= '/run/media/lorenzo/My Passport/Data/'
path = data.PatientExperiment.unique()[0]
path = path.replace('\\','/')
ISI_data_path = experiments_path + path

os.chdir(ISI_data_path)

Times_files_name_list = glob.glob('times_NSX*.mat')

df = load_times_files('times_NSX2.mat')
classes = set(df.cluster)


plt.close('all')
class_1 = df[df.cluster == 1]
xa =  np.diff(class_1.time)
hist = np.histogram(xa,bins = 100,range = {0,100})
plt.bar(hist[1][:-1],hist[0])
plt.legend()