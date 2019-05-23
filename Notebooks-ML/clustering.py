#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import libraries

# linear algebra
import numpy as np 
# data processing
import pandas as pd 
# data visualization
from matplotlib import pyplot as plt 
from math import sqrt


# In[10]:


def calculate_initial_centers(dataset, k):
    centroids = []
    n = dataset.shape[1]
    minimos = np.min(dataset,  axis = 0)
    maximos = np.max(dataset, axis = 0)
    for i in range(0, k):
        centroids.append(np.random.uniform(minimos, maximos, n))
    return np.array(centroids)


# In[11]:


def euclidian_distance(a, b):
    return sqrt(np.sum([(i - j)**2 for i, j in zip(a,b)]))


# In[12]:


def distance_manhattan(x1, x2):
    return np.sum([abs(i-j) for i, j in zip(x1,x2)])


# In[13]:


def nearest_centroid(a, centroids):
    dist = [euclidian_distance(a, centro) for centro in centroids]
    for i in range(0, len(dist)):
        if dist[i] == min(dist):
            nearest_index = i
            break
    return nearest_index


# In[14]:


def all_nearest_centroids(dataset, centroids):
    nearest_indexes = []
    for row in dataset:
        nearest_indexes.append(nearest_centroid(row, centroids))
    return np.array(nearest_indexes)


# In[15]:


def inertia(dataset, centroids, nearest_indexes):
    inertia = np.sum([euclidian_distance(dataset[i], centroids[nearest_indexes[i]])**2 for i in range(0, len(dataset))])
    return inertia


# In[16]:


def update_centroids(dataset, centroids, nearest_indexes):
    dimensao = len(np.unique(nearest_indexes))
    soma = np.zeros((dimensao, dataset.shape[1]))
    total = np.zeros(dimensao)
    for i in range(0, len(dataset)):
        soma[nearest_indexes[i]] += dataset[i]
        total[nearest_indexes[i]] += 1
    
    centroids = [soma[i]/total[i] for i in range(0, dimensao)]
    return np.array(centroids)


# In[17]:


def kmeans(X, K, distance_metric=None, max_iter=None, num_rep=None):    
    cluster_index = []
    centroids = []
    inertias = []
    
    if max_iter is None:
        max_iter = 10
    else:
        max_iter = max_iter
        
    if distance_metric is None:
        distance_metric = 'euclidean'
    else:
        distance_metric = distance_metric
        
    if num_rep is None:
        num_rep = 100
    else:
        num_rep = num_rep
        
    for index in range(0, num_rep):
        cluster_centers = calculate_initial_centers(X, K)
        labels = all_nearest_centroids(X, cluster_centers)
        old_inertia = inertia(X, cluster_centers, labels)
        inertia_ = [] 
        inertia_.append(old_inertia)
        for inter in range(0, max_inter):
            cluster_centers = update_centroids(X, K, labels)
            labels = all_nearest_centroids(X, cluster_centers)
            inertia_.append(inertia(X, cluster_centers, labels))
        cluster_index.append(labels)
        centroids.append(cluster_centers)
        inertias.append(inertia_)
    
        indice = np.argmin(inestias[:,-1])
    
    return {'cluster_index': cluster_index[indice], 'centroids': centroids[indice], 'loss': inertias[indice]}

def predict(X, cluster_centers):
    return all_nearest_centroids(X, cluster_centers)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




