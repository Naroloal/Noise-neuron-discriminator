#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib.cm as cm
from Funciones_auxiliares import Mat_to_dataframe,see_teams,get_cluster_indexs,plot_branch,plot_teams
from scipy import cluster
import scipy as sp
import math
from scipy.spatial.distance import squareform
import itertools
import chart_studio.plotly as plotly


# In[2]:


plt.close('all')
'''Mat file to dataframe for working in python'''
NOISE = pd.read_pickle('NOISE')
NEURON = pd.read_pickle('NEURON')
NOISE_MAYBE = pd.read_pickle('Noise_Maybe')


# In[3]:


df = pd.concat([NOISE,NEURON],ignore_index = True,sort = False)
lenght = len(df.Mean)


# In[4]:


''''Matrix Initialization'''
Mean = []
for i in range(len(df.Mean)):
    Mean.append(df.Mean[i])


# In[5]:


Correlation_p = np.zeros((lenght,lenght))
P_Value_p = np.zeros((lenght,lenght))


# In[6]:


'''Pearson tests'''
for i in range(lenght):
    for j in range(lenght):
        Correlation_p[i,j],P_Value_p[i,j] = stats.pearsonr(df.Mean[i],df.Mean[j])


# In[36]:


threshold2 = 0.9## For Pearson is defined as 1 - correlation


# In[15]:

'''Correlation matrix plot'''
plt.imshow(Correlation_p)


# In[35]:

'''Correlation between good an bad spikeshapes'''
Good_Bad_correlation = Correlation_p[len(NOISE)+1:,:len(NOISE)]
#plt.imshow(Good_Bad_correlation)


# In[40]:


mixed = np.where(Good_Bad_correlation > threshold2)


# In[41]:

'''Plots of good with bads'''
fig = plt.figure(0)
axes = fig.subplots(7,4)
axes_flt = axes.flatten()
for i in range(len(mixed[0])):
    axes_flt[i].plot(df.Mean.iloc[len(NOISE)+1 +mixed[0][i]])
    axes_flt[i].plot(df.Mean.iloc[mixed[1][i]])
    axes_flt[i].title.set_text('spikes = ({},{}), Correlation= {}'.format(len(NOISE) + 1 + mixed[0][i],mixed[1][i],round(Correlation_p[len(NOISE) + 1 + mixed[0][i],mixed[1][i]],3)))


# In[ ]:

def plot_mixed_clusters(i,j,df,axes_flt,r,plot_good = False):
    X1 = df.Bulk.iloc[i]
    X1_m = df.Mean_nn.iloc[i]
    X2 = df.Bulk.iloc[j]
    X2_m = df.Mean_nn.iloc[j]
    if plot_good:
        for k in range(len(X1)):  
            axes_flt[0].plot(X1[k],'b',linewidth = 0.1)
        axes_flt[0].plot(X1_m,'k')
    for k in range(len(X2)):
        axes_flt[r].plot(X2[k],'r',linewidth = 0.1)
        axes_flt[r].plot(X2_m,'k')
    fig.suptitle('spikes ({},{})'.format(i,j))


elements,counts = np.unique(mixed[0],return_counts=True)
d = dict(zip(elements, counts))
# In[]:

num_fig = 1
num_axes = 1
plot_good = True
fig = plt.figure(num_fig,figsize = (5,10))
axes = fig.subplots(d[mixed[0][0]] + 1,1)
axes_flt = axes.flatten()
for k in range(len(mixed[0])):
    if (k!=0) and mixed[0][k]!= mixed[0][k-1]:
        num_fig +=1
        fig = plt.figure(num_fig,figsize=(5,10))
        axes = fig.subplots(d[mixed[0][k]] + 1,1)
        axes_flt = axes.flatten()
        num_axes = 1
        plot_good = True
    plot_mixed_clusters(mixed[0][k]+len(NOISE) + 1,mixed[1][k],df,axes_flt,num_axes,plot_good=plot_good)
    num_axes+=1
    plot_good = False
# In[]    
good_mixed = [len(NOISE)+1 + elements[i] for i in range(len(elements))]
bad_mixed = np.unique(mixed[1])
Total = np.concatenate((good_mixed,bad_mixed))

# In[]    

Num_plots = len(d.keys()) + len(mixed[1])
Num_figs = Num_plots//25 + 1
List_figs = [plt.figure() for i in range(Num_figs)]
Axes_flt = [fig.subplots(5,5).flatten() for fig in List_figs]
# In[]    


for i in range(len(Total)):
    ax_to_plot = i%25
    fig_to_plot = i//25
    data = df.Bulk.iloc[Total[i]]
    color = df.bNoise.iloc[Total[i]]
    if color == 1:
        color = 'r'
    else:
        color = 'b'
    data_m = df.Mean_nn.iloc[Total[i]]
    Axes_flt[fig_to_plot][ax_to_plot].plot(data_m[i],color = 'k', label = 'spike {}'.format(Total[i]),linewidth = 1)
    Axes_flt[fig_to_plot][ax_to_plot].title.set_text('spike = {}'.format(Total[i]))
    for j in range(len((data))):   
        Axes_flt[fig_to_plot][ax_to_plot].plot(data[j],color = color,linewidth = 0.1)
        

# In[]
'''Rodrigo quiere que tiremos todos los ruidos'''
df.drop(bad_mixed,inplace = True)
df.reset_index(inplace = True)
df.drop(columns = 'index',inplace = True)
New_treshold = 0.95
# In[]

df_clean = pd.DataFrame(np.array(df['Mean'].tolist()).transpose())
Corr = df_clean.corr()
Corr = Corr.values
Gb_Corr = Corr[169:,:169]

Mixed_gb = np.where(Gb_Corr > New_treshold)
print(Mixed_gb)

df[df.bNoise == 1].to_pickle('FINAL_NOISE')
