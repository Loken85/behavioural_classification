# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:50:42 2019

@author: proxy_loken


Script for running automated behavioural state tracking on a deeplabcut feature
dataset

"""

import pandas
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import b_utils

import b_tracking_parameters
import iterative_smoothing

import pywt
import wavelets
import t_sne
import watershed_t




"""
PART 1: File read and pre-processing

Read in a deeplabcut feature dataset and perform smoothing/etc. as necessary

TODO: keep filepath and name stem as variables for later use in defining savepaths

"""

# Read in file
file = pandas.read_hdf('cut_ECG10_03-15_stackDeepCut_resnet50_stackJun28shuffle1_500000.h5')
# Trim top-level index
parts = file['DeepCut_resnet50_stackJun28shuffle1_500000']
# Sample frequency (fps of the camera in most cases)
sfreq = 30
# Dimensionality of DLC features (multiple camera views)
inDims = 3
# feature to align projections (default shoulder girdle SG)
a_ft = 'SG'
# total number of features across all cameras
L = len(parts.columns.levels[0])
# actual number of features
n_ft = L/(inDims-1)

# Optional: list of feature names: only necessary if inDims is >=3 or if you
# want to exclude some features
feats = ['N','SG','LP','RP','PV','LR','RR','TB','TM'] 
print('Tracked features:' + str(feats))


print('Smoothing Data')
smooth_data = iterative_smoothing.iter_smooth(parts)


# Create projections dataframe and align to a chosen feature
# TODO: get the second dimension working. Currently only processes dim 1
dims = 1
while dims < inDims:
    cfeats = [f + str(dims) for f in feats]
    rawProjections = b_utils.create_raw_projections(smooth_data,cfeats)
    alignProjections = b_utils.align_projections2D(rawProjections,a_ft,dims)



# TODO: Plot aligned features through time
    


"""
PART 2: Wavelet Transform and embedding

Calculate Morlet wavelets over the frequency range, and embed the aigned
features in high dimensional wavelet space

"""
# calculate freqs and scales for wavelet transform
scales, freqs = wavelets.calculate_scales(b_tracking_parameters.minF,b_tracking_parameters.maxF,sfreq,b_tracking_parameters.numPeriods)

scalogram, scalonp = wavelets.wavelet_transform(alignProjections, scales, freqs, sfreq)


# example plot feature from scalogram. needs to be a numpy array
amps = scalogram['N1']['x'].T
amps = amps.to_numpy()
b_utils.plot_wamps(amps,freqs,'N1 x Amplitudes')



"""
PART 3: Dimensionality Reduction and Clustering

T-SNE to reduce dimensions and cluster. Generate a density map based on a gaussian kernel,
separate and name clusters based on neighbourhood estimation (watershed trasform)

"""
# perform t-sne embedding. (Takes dims and perplexity as optional arguments)
t_out = t_sne.tsne_embedding(scalonp)


# get the gaussian kernel density of the tsne space. GRid x and y are for plotting in the tsne space
density, grid_x, grid_y = t_sne.calc_density(t_out)

# perform the watershed transform to label clusters in the tsne space
labels = watershed_t.watershed_transform(density, 10, peaks=True)

watershed_t.plot_labels(labels)

# label each frame with the cluster to which it belongs
l_frames = watershed_t.label_frames(t_out, labels)

