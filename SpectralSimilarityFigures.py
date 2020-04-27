#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import pyteomics.mzxml as mzxml
import numpy as np

import scipy.interpolate
import os


# In[2]:


def snip_baseline_removal(original,width = 50):
    """ Estimate, and then remove, the baseline from a data-set using the SNIP algorithm. """
    # First, perform the non-linear stage
    m = len(original)
    data = np.log1p(np.log1p(np.sqrt(1.0 + np.abs(original))))

    for k in range(width):
        shifted = np.roll(data,k) + np.roll(data,-1 * k)
        shifted[0:k] = data[0:k] + data[k:2 * k]
        shifted[m - k : m] = data[m - k : m] + data[m - 2 * k: m - k]
        data = np.minimum(0.5 * shifted, data)
        
    # Undo the non-linear stage, and subtract the resulting estimate, and force positivity
    adjusted = original - (np.square(np.expm1(np.expm1(data))) - 1.0)
    least = min(adjusted)
    if least < 0:
        adjusted -= least
    return adjusted + 1
    
def all_valid_mzxml(directory, low = 1010, high = 2390, points = 8000):
    grid = np.linspace(low,high, points)
    
    for i in os.listdir(directory):
        if i[-5:] != 'mzXML':
            continue
        
        for x in mzxml.read(os.path.join(directory,i)):
            mz, inten = x['m/z array'], x['intensity array']
            if min(mz) > low or max(mz) < high:
                print("Skipping")
                continue
            snipped_and_resampled = scipy.interpolate.griddata(mz, snip_baseline_removal(inten), grid)
            snipped_and_resampled /= np.trapz(snipped_and_resampled)
            
            yield i.split(';')[0], i, snipped_and_resampled
        
def group_by_label(gen):
    labels = dict()
    
    for label, file, spectra in gen:
        if label not in labels:
            labels[label] = []
            
        labels[label].append((file, spectra))
    return labels
    
datasets = dict()
for i in os.listdir("data"):
    datasets[i] = group_by_label(all_valid_mzxml(os.path.join('data',i)))


# In[3]:


def matrix(comp, a,b):
    ka, kb = list(sorted(a.keys())), list(sorted(b.keys()))
    kb.reverse()
    
    output = np.empty((len(ka), len(kb)))
    for ia, aa in enumerate(ka):
        for ib, bb in enumerate(kb):
            output[ia,ib] = comp(a[aa], b[bb])
            
    return output, ka, kb


def average_dot_product(p1, p2):
    n = 0
    acc = 0
    
    for a in p1:
        a = a[1]
        for b in p2:
            b = b[1]
            acc += a.dot(b) / np.sqrt(a.dot(a) * b.dot(b))
            n += 1
    return acc / n


# In[4]:


def matrix_as_bubble(mat, k1, k2, l1, l2):
    
    def purge_underscores(l):
        return list(''.join(x.split('_')) for x in l)
    
    n,m = mat.shape
    x = np.empty(n * m)
    y = np.empty_like(x)
    s = np.empty_like(x)
    k = 0
    
    for i in range(n):
        for j in range(m):
            x[k] = i
            y[k] = j
            s[k] = mat[i,j]
            k += 1
    
    
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    
    for j in range(k):
        sj = s[j]
        if sj < 0.40:
            continue
        plt.text(x[j], y[j], str(sj)[0:4], fontsize=int(14 * sj), horizontalalignment='center'
                 ,verticalalignment='center', color='white')
    
    size = fig.get_size_inches()*fig.dpi
    scale = 0.9 * (3.1415 / 4) * min(size[0]/n, size[1]/m) ** 2
    
    plt.scatter(x,y, s * scale)
    plt.xticks(range(n), purge_underscores(k1), rotation = 'vertical')
    plt.yticks(range(m), purge_underscores(k2))
    plt.ylabel(l2, fontsize = 20)
    plt.xlabel(l1, fontsize = 20)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    


# In[5]:


for name1, ds1 in datasets.items():
    for name2, ds2 in datasets.items():

        m, k1, k2 = matrix(average_dot_product, ds1, ds2)
        matrix_as_bubble(m, k1, k2, name1, name2)

