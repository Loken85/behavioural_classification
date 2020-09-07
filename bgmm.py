#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:55:29 2020
@author: proxy_loken


Bayesian gaussian mixture functions                 


"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

from scipy import linalg
import matplotlib as mpl
import itertools



def estimate_bgmm(t_out, max_components=20):
    
    bgmm = BayesianGaussianMixture(n_components=max_components, covariance_type='full', n_init=5, init_params='kmeans').fit(t_out)
    
    bgmm_predict = bgmm.predict(t_out)
    
    plot_results(t_out, bgmm_predict, bgmm.means_, bgmm.covariances_, 1, 'Bayesian Gaussian Mixture Model of Estiamte of Clusters')
    
    return bgmm, bgmm_predict




def plot_results(X, Y_, means, covariances, index, title):
    
    colour_iter = itertools.cycle(['navy','c','cornflowerblue','gold','darkorange','red','green','purple','magenta','yellow','gray','teal'])
    fig = plt.figure()
    
    plt.subplot(2,1,1 + index)
    for i, (mean, covar, colour) in enumerate(zip(means,covariances,colour_iter)):
        
        v,w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        # remove the redundent components before plotting
        
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i,1], .8, color=colour)
        
        # plot an ellipse to show the gaussian component
        
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi # convert to degrees
        
        # plot elipsoids for each cluster: optional
        
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=colour)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)
        
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
    
    
    