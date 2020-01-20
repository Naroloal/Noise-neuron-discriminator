#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 07:44:12 2020

@author: lorenzo
"""


from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib.cm as cm
from scipy import cluster
import scipy as sp
import itertools


def get_cluster_indexs(number,fl):
    if isinstance(number,list):
        indexes_list = []
        for i in number:
            indexes_list.append(np.where(fl == i)[0])
        return indexes_list
    else: return np.where(fl == number)[0]

def plot_branch(cluster,fl,s_plot,df,label = False):
    tup = get_cluster_indexs(cluster,fl)
    spikes = df.Mean[tup]
    labels1 = list(df.bNoise[tup])
    labels2 = list(df.bUnSure[tup])
    i = 0
    for element in spikes:
        if label:
            s_plot.plot(element,label = '{},{}'.format(labels1[i],labels2[i]))
            i+=1
            s_plot.legend()
        else:s_plot.plot(element)
    return tup

def Mat_to_dataframe(path):
    Data = loadmat(path)
    Data = Data['NoiseClusters']
    Data = Data[0]
    Column_names = Data.dtype.names
    Dataframe = {}
    for i in range(len(Column_names)):
        Dataframe[Column_names[i]] = []
    for i in range(len(Data)):
        row = Data[i]
        for j in range(len(row)):
            column = row[j]
            Dataframe[Column_names[j]].append(column)
    Data = pd.DataFrame.from_dict(Dataframe)
    return Data

def see_teams(lista_de_teams,fl):
    cmap = plt.get_cmap('gnuplot')
    if isinstance(lista_de_teams,list):
        colors = [cmap(i) for i in np.linspace(0, 1, len(lista_de_teams))]
        speiksheipes = get_cluster_indexs(lista_de_teams,fl)
        fig = plt.figure(6)
        i = 0
        for element in speiksheipes:
            for element2 in element:
                plt.plot(df.Mean[int(element2)],c = colors[i])
                i+=1
    else:
        speiksheipes = get_cluster_indexs(lista_de_teams,fl)
        colors = [cmap(i) for i in np.linspace(0, 1, len(speiksheipes))]
        fig = plt.figure(6)
        i = 0
        for element in speiksheipes:
            plt.plot(df.Mean[int(element)],c = colors[i])
            i+=1
            
def dataframe_to_math(path,df):
    a = {name: col.values for name, col in df.items()}
    sp.io.savemat(path,a)
        
    
def Search_candidates(i,spikes,fl,N_teams,Correlation_p,size,Good_candidates = False):
    r = []
    Not_good_candidate = []
    for Candidates in itertools.combinations(spikes,r = size):
            while not Good_candidates:
                Good_candidates = True
                for j in range(1,N_teams+1):
                    if not j==i:
                        other_spikes = get_cluster_indexs(j,fl)
                        for k in range(len(other_spikes)):
                            check = [Correlation_p[i][k] > 0.9 for q in Candidates]
                            if not all(check):
                                Good_candidates = False
                                break
                            if not Good_candidates:break
                    
                if Good_candidates: r.append([i,Candidates])         
                else:
                   print('No good candidates found for Team {}'.format(i))
                   Good_candidates = True
                   Not_good_candidate.append(i)
    return [r,Not_good_candidate]
        