# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""


import pandas as pd
import numpy as np

''''Dado los clusters corregidos por Rodrigo, este codigo toma los nombres en 
noise_to_neuron y neuron_to_noise, para cambiar el label de la base de datos Data_all_sessions.
Ademas elimina la columna bUnSure ya que a partir de aca se tiene una base lista y revisada para trabajar.
Notar que eso no implica que este libre de inconsistencias, ya que la eleccion de buena/mal cluster es muy subjetiva
y poco arbitraria'''

a = pd.read_pickle('Datos/Data_all_sessions')
                   
noise_to_neuron = []
neuron_to_noise = []

with open('noise_to_neuron','r') as f:
    for line in f:
        noise_to_neuron.append(line.strip())
        
with open('neuron_to_noise','r') as f:
    for line in f:
        neuron_to_noise.append(line.strip())

names = {}
'''Obtenemos la conversion de patientExperiment y lo metemos en un dataframe'''
with open('names.txt','r') as f:
    for line in f:
        line = line.split(',')
        names[line[1].strip()] = line[0].strip()
        
'''Definiomos al funcion que encuentra el patientexp,channel,cluster dado un label'''
def find_PE_Ch_Cl(string):
    PE,Ch,Cl =  string.split('_')
    patientExperiment = names[PE]
    return patientExperiment,int(Ch),int(Cl)

'''Primero cambiamos noise to neuron segun sea necesario'''
for label in noise_to_neuron:
    PE,ch,cl = find_PE_Ch_Cl(label)
    aux = a.loc[(a.PatientExperiment == PE)& (a.Channel == ch) & (a.Cluster == cl)]
    if len(aux) == 0: print(label,'no existe en la base de datos')
    elif aux.bNoise.iloc[0] == 0: print(label, 'ya esta labeleada como neurona')
    else:
        a.loc[(a.PatientExperiment == PE)& (a.Channel == ch) & (a.Cluster == cl),'bNoise'] = 0
        
'''segundo cambiamos neuron to noise segun sea necesario'''
for label in noise_to_neuron:
    PE,ch,cl = find_PE_Ch_Cl(label)
    aux = a.loc[(a.PatientExperiment == PE)& (a.Channel == ch) & (a.Cluster == cl)]
    if len(aux) == 0: print(label,'no existe en la base de datos')
    elif aux.bNoise.iloc[0] == 1: print(label, 'ya esta labeleada como noise')
    else:
        a.loc[(a.PatientExperiment == PE)& (a.Channel == ch) & (a.Cluster == cl),'bNoise'] = 1
    
        
    



