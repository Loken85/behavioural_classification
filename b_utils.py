# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:48:26 2019

@author: adria

Utilities for automated behavioural state tracking with deeplabcut input
"""

import pandas
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter


# Turns a dataframe into a dictionary of numpy arrays
# one (length X 3) array for each part
def frame_to_darray(data):
    darray = {}
    
    for feature in list(data.T.index.get_level_values(0)):
        darray[feature] = data[feature].to_numpy()        
    
    
    return darray

# saves a .mat file for use in matlab. Takes in a dataframe, saves as a 
# struct containing a set of numpy arrays
# Currently saves to PWD, TODO: set the filepath dynamically
def expdf_to_matlab(data):
    arry = frame_to_darray(data)
    
    sio.savemat('data_mat.mat', {'data': arry})
    

# same as above, but takes in dict or numpy array
# could be combined with a type query...but whatever
def expnp_to_matlab(data):
    
    sio.savemat('data_mat.mat', {'data': data})
    
    
# create a raw projections dataframe. extracts just x and y positions for features
# in feats list
def create_raw_projections(data,feats):
    
    data = data.T
    data = data.sort_index()
    rproj = data.loc[(feats,('x','y')),:]
            
    rproj = rproj.T    
    return rproj

# aligns the projections to a selected feature. Selected feature positions will
# be all zeros 
def align_projections2D(rproj,a_ft,dim):
    aproj = rproj.copy()
    # assemble the full feature lable
    feature = a_ft + str(dim)
    aproj = aproj.subtract(aproj[feature], level=1)
    
    return aproj

# redundant
def align_projections3D(rproj,a_ft,n_ft):
    
    return aproj


# plots the wave aplitudes of a single feature projected into wavelet space
# TODO: add figure name and axes labels
def plot_wamps(scalo,freqs,name, figs_per_row=5):
    # number of subplots
    num_plts = np.size(scalo,0)
    # number of rows 
    n_rows = np.ceil(num_plts/figs_per_row)
    n_rows = n_rows.astype(int)
    # create suplots. set x and y axes to be shared, and set spacing
    #fig, axs = plt.subplots(n_rows,figs_per_row,sharex=True,sharey=True,gridspec_kw={'hspace': 0,'wspace' : 0})
    # version without shared axis
    fig, axs = plt.subplots(n_rows,figs_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig.suptitle(name)    
    # only use outer axes labels
    for ax in axs.flat:
        ax.label_outer()
    
    for r in range(0,n_rows):
        for n in range(0,figs_per_row):
            curr = r*figs_per_row + n
            if curr<num_plts:
                axs[r,n].plot(scalo[curr,:])
            else:
                break
    
    
# generates and plots a graphical object showing the current frame location in the
# clustered reduced dimensionality space
def plot_curr_cluster(t_out, labels, frame, xi, yi):
    
    # create figure
    fig, ax = plt.subplots()
    # plot the clusters
    plt.pcolormesh(xi, yi, labels)
    # plot location of current frame (x, y reversed because t_out is transposed)
    y, x = t_out[frame]
    plt.scatter(x, y, s=10, c='red', marker='+')
    
    plt.show()
    


# generates and saves a movie of the t-sne space locations for each frame
def cluster_anim(t_out, labels, xi, yi, fps, start_f = 0, end_f = 1000):
    metadata = dict(title="T-sne Space Plot", artist="matplotlib", comment="Movie of t-sne space locations for each frame")
    
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    
    fig, ax = plt.subplots()
    
    #plt.xlim(np.min(xi)-5,np.max(xi)+5)
    #plt.ylim(np.min(yi)-5,np.max(yi)+5)
    
    #frames = np.size(t_out, 0)
    frames = end_f-start_f
    
    with writer.saving(fig, "location_plot.mp4", frames):
        for i in range(start_f,end_f):
            plt.pcolormesh(xi, yi, labels)
            
            ax.autoscale(False)
            
            y, x = t_out[i]
            plt.scatter(x,y,s=10, c='red', marker='+')
            
            writer.grab_frame()
            
            
# helper function for finding the index of the nearest value in an array
# note: this will be slow on large arrays
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_ethogram(l_frames):
    #max_l = np.max(l_frames)
    frames = np.size(l_frames,0)
    
    #ys = range(0,max_l)
    xs = range(0, frames)
    
    fig,ax = plt.subplots()
    
    plt.scatter(xs, l_frames, c=l_frames, s=10, cmap='jet')
    
    plt.show()
    
# count number of occurances of labels in label array 
def count_labels(l_array):
    labels, counts = np.unique(l_array, return_counts=True)
    
    return labels, counts

# count consecutive occurances of labels in label array
def count_consecutive_labels(l_array):
    l_array = l_array.astype(int)
    counts_list = []
    
    for i in range(0,np.max(l_array)):
        
        bool_arr = l_array==i
        count = np.diff(np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0])[::2]
        counts_list.append(count)
        
    return counts_list
    
# trim consecutive count arrays to discard short "behaviours"
def trim_counts(counts_list,threshold=5):
    
    trim_counts = []
    
    for count in counts_list:
        
        count = np.sort(count)
        
        inds = np.where(count>=threshold)
        trim = count[inds]
        
        trim_counts.append(trim)
        
    return trim_counts


# plot a set of counts as histograms
def plot_label_counts(counts_list, plots_per_row=3, name='Label Counts',color='blue'):
    
    max_count = 0
    
    for count in counts_list:
        if count.any():
            curr_max = np.max(count)
        if curr_max > max_count:
            max_count = curr_max
    
    num_plots = len(counts_list)
    n_rows = np.ceil(num_plots/plots_per_row)
    n_rows = n_rows.astype(int)
    
    bins = range(0,max_count+1)
    
    
    fig, axs = plt.subplots(n_rows,plots_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig.suptitle(name)    
    # only use outer axes labels
    for ax in axs.flat:
        ax.label_outer()
    
    for r in range(0,n_rows):
        for n in range(0,plots_per_row):
            curr = r*plots_per_row + n
            if curr<num_plots:
                axs[r,n].hist(counts_list[curr],bins=bins,color=color)
            else:
                break
