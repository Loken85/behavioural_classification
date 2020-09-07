#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 05:46:00 2020

@author: seamans

Working script for neural predicted annotations -> behavioural classification testing


"""

import scipy.io
import numpy as np

import matplotlib.pyplot as plt

import b_utils
import bgmm

import pickle

import b_tracking_parameters
import iterative_smoothing

import pywt
import wavelets
import t_sne
import watershed_t


# .mat files containing true and predicted pose data
raw_file = '03_30_I_Blocks231_cat_data.mat'
pred_file = '03_30_I_Blocks231_CNN_MVMT_preds2.mat'

# Version for loading matlab three-tone sets
# Load in data set from .mat file
# Modify here to change the loaded matrices
def load_data_set():
    mat = scipy.io.loadmat(raw_file)
    #Stmtx = mat['STbintrim']
    Stmtx = mat['STbintrim']    
    Stmtx = np.transpose(Stmtx)
    #gridrefs = mat['gridrefs']
    #comdels = mat['comdels']
    #comtrim = mat['comtrim']
    #factors = mat['red_dists']
    factors = mat['red_dists']
    
    return Stmtx, factors


def load_pose_data():
    mat = scipy.io.loadmat(pred_file)
    raw_poses = mat['Y_train']
    pred_poses = mat['train_preds']
    
    return raw_poses, pred_poses








# Load Pose data

raw_poses, pred_poses = load_pose_data()


# misc parameters

# Sample frequency (fps of the camera in most cases)
sfreq = 5



# apply wavelets to pose data
"""
PART 2: Wavelet Transform and embedding

Calculate Morlet wavelets over the frequency range, and embed the aigned
features in high dimensional wavelet space

"""
# calculate freqs and scales for wavelet transform
scales, freqs = wavelets.calculate_scales(b_tracking_parameters.minF,b_tracking_parameters.maxF,sfreq,b_tracking_parameters.numPeriods)

scalonp = wavelets.wavelet_transform_np(raw_poses, scales, freqs, sfreq)


# example plot feature from scalo array. needs to be a numpy array

amps = scalonp[0:24,:]
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
l_frames = watershed_t.label_frames(t_out, labels, grid_x, grid_y)


# alternate cluster labelling with bgmm

bgmmm, bgmm_predict = bgmm.estimate_bgmm(t_out)


# watershed ethogram plot


b_utils.plot_ethogram(l_frames)

# bgmm ethogram plot

b_utils.plot_ethogram(bgmm_predict)



# Pickle data for video plotting if desired

xi = grid_x
yi = grid_y

data = {'t_out': t_out, 'labels': labels, 'l_frames': l_frames, 'bgmm_predict': bgmm_predict, 'xi': xi, 'yi': yi}

pickle.dump(data, open('i231_full_data.p','wb'))


# Unpickling
with open('pred_data.p','rb') as fp:
    l_data = pickle.load(fp)    

    t_out = l_data['t_out']
    labels = l_data['labels']
    l_frames = l_data['l_frames']
    xi = l_data['xi']
    yi = l_data['yi']



# Divvy up into blocks, tones, etc for analysis
    
neut_l_frames = l_frames[18229:24811]
shock_l_frames = l_frames[9974:16505]
food_l_frames = l_frames[1627:8206]

neut_bgmm = bgmm_predict[18229:24811]
shock_bgmm = bgmm_predict[9974:16505]
food_bgmm = bgmm_predict[1627:8206]


neut_labels,neut_counts = b_utils.count_labels(neut_l_frames)
shock_labels,shock_counts = b_utils.count_labels(shock_l_frames)
food_labels,food_counts = b_utils.count_labels(food_l_frames)

neut_blabels,neut_bcounts = b_utils.count_labels(neut_bgmm)
shock_blabels,shock_bcounts = b_utils.count_labels(shock_bgmm)
food_blabels,food_bcounts = b_utils.count_labels(food_bgmm)

# plot raw counts
plt.figure()
plt.hist(shock_labels,weights=shock_counts,color='red')
plt.figure()
plt.hist(food_labels,weights=food_counts,color='green')
plt.figure()
plt.hist(neut_labels,weights=neut_counts,color='blue')


plt.figure()
plt.hist(shock_blabels,weights=shock_bcounts,color='red')
plt.figure()
plt.hist(food_blabels,weights=food_bcounts,color='green')
plt.figure()
plt.hist(neut_blabels,weights=neut_bcounts,color='blue')


# consecutive counts

neut_ccounts = b_utils.count_consecutive_labels(neut_l_frames)
shock_ccounts = b_utils.count_consecutive_labels(shock_l_frames)
food_ccounts = b_utils.count_consecutive_labels(food_l_frames)

neut_bccounts = b_utils.count_consecutive_labels(neut_bgmm)
shock_bccounts = b_utils.count_consecutive_labels(shock_bgmm)
food_bccounts = b_utils.count_consecutive_labels(food_bgmm)

# plot consecutive counts
b_utils.plot_label_counts(shock_ccounts,plots_per_row=4,name='Shock Consecutive Counts',color='red')
b_utils.plot_label_counts(food_ccounts,plots_per_row=4,name='Food Consecutive Counts',color='green')
b_utils.plot_label_counts(neut_ccounts,plots_per_row=4,name='Neutral Consecutive Counts',color='blue')

b_utils.plot_label_counts(shock_bccounts,plots_per_row=4,name='Shock Consecutive Counts',color='red')
b_utils.plot_label_counts(food_bccounts,plots_per_row=4,name='Food Consecutive Counts',color='green')
b_utils.plot_label_counts(neut_bccounts,plots_per_row=4,name='Neutral Consecutive Counts',color='blue')

