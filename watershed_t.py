# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:58:20 2019

@author: proxy_loken

Watershed transform functions

"""



import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


# performs the watershed transform on a density map. INPUTS density: and array of
# densities to be used as heights for the watershed transform.compact: compactness
# of the watersheds (use 0 if not using compact transform). peaks: boolean, whether to
# use peak_local_max to precalculate local maxima OUTPUTS: watershed labels, same size
# as density input 
def watershed_transform(density, compact, peaks=False):
    
    mask = mask_density(density)
    
    if peaks:
        l_maxs = peak_local_max(density,indices=False)
        markers = ndi.label(l_maxs)[0]
        labels = watershed(-density, markers, mask=mask, compactness = compact)
    else:
        labels = watershed(-density, mask=mask, compactness=compact)
    
    return labels
    

def smooth_density(density, epsilon=0):
    #TODO if necessary
    return 0
    

def mask_density(density, epsilon=0):
    
    mask = density > np.max([np.std(density, epsilon)])
    
    return mask

def plot_labels(labels):
    # TODO: define axis, colors, etc. 
    plt.imshow(labels)
    


# helper function for finding the index of the nearest value in an array
# note: this will be slow on large arrays
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

    
# labels each frame in t_out with the cluster number from the watershed transform
def label_frames(t_out, labels, xi, yi):
    # TODO add option for 1-hots encoding
    l_frames = np.zeros(np.size(t_out,0))
    x, y = np.transpose(t_out)
    
    for i in range(np.size(t_out,0)):
        x_idx = find_nearest_index(xi[:,0], x[i])
        y_idx = find_nearest_index(yi[0,:], y[i])
        l_frames[i] = labels[x_idx,y_idx]
        
    return l_frames


    
    