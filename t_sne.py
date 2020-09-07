# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:24:08 2019

@author: proxy_loken

T-distributed stochastic neighbourhood estimation functions 


"""

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.kde import gaussian_kde

# fit scalogram(wavelet embedded) data into a reduced dimensionality space using
# T-sne. INPUT: dims x data points numpy array. dims: number of output dimensions
# perplexity: approx. number of NN in the projected space
# can add other parameters (learning rate, iterations, etc. if necessary)
def tsne_embedding(scalos, dims=2, perplexity=30):
    # TODO: pull arguments from parameters file
    tsne_out = TSNE(n_components=dims,perplexity=perplexity).fit_transform(np.transpose(scalos))
    
    return tsne_out


# plots the tsne output as a map of a 2d histogram. INPUTS: t_out: 2d array of
#coordinates for each data point. dense: if false, plots sample count for each 
# bin, if true, counts probability density. points: number of bins for histogram
def plot_density_2dhist(t_out, dense, points):
    #separate dimensions
    x, y = np.transpose(t_out)
    # make 2d histogram
    z, xedges, yedges = np.histogram2d(x,y,density=dense,bins=points)
    # plot the resulting map
    
    plt.pcolormesh(xedges, yedges, z.T)
    


# plots the tsne output as a map of a 2d gaussian kernal probability density
# estimate. INPUTS: t_out: 2d array of coordinates for each data point
def plot_density_gkernel(t_out):
    #calculate the density
    density,xi,yi = calc_density(t_out)
    #plot the resulting map
    plt.figure()
    plt.pcolormesh(xi, yi, density)
    

# calculates the density(probability) of the tsne output using a gaussian kernel 
def calc_density(t_out):    
    # separate dimensions
    x,y = np.transpose(t_out)
    # calculate kernal density estimate
    k = gaussian_kde(np.vstack([x, y]))
    # make mesh to hold values (uses default values for mesh size(sqrt numsamples / 2) change if necessary)
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    # flatten 3d kernal into density across the grid
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # reshape to an array and return
    return zi.reshape(xi.shape),xi,yi
    