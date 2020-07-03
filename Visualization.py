#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:14:21 2020

@author: lorenzo
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap as ISO
from sklearn.cluster import DBSCAN
    

Data = pd.read_pickle('Data_to_analize')
Color = Data.Team.to_list()
    

df = pd.DataFrame(np.array(Data['Mean'].tolist()).transpose())
Corr = df.corr()

D = 1-Corr

pca = PCA(3)
D_trans = pca.fit_transform(D)


#mds = MDS(3)
#D_trans= mds.fit_transform(D)

#iso = ISO(n_components=3)
#D_trans =  iso.fit_transform(D)


plot_data(D_trans)



def plot_data(D,color = Color):
    dim = len(D[0])
    print(dim)
    if (dim !=2) and (dim!=3): raise ValueError('Dim incorrecta')
    if dim == 2:
        X = [i[0] for i in D]
        Y = [i[1] for i in D]
        fig = plt.figure(4)
        plt.scatter(X,Y,c = Color,cmap = 'gnuplot')
    if dim == 3:
        X = [i[0] for i in D]
        Y = [i[1] for i in D]
        Z = [i[2] for i in D]
        fig = plt.figure(5)
        ax = Axes3D(fig)
        ax.scatter(X,Y,Z,c = Color,cmap = 'gnuplot')
        