# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:36:48 2019

@author: proxy_loken


Parameter set for automated behavioural state tracking

Contains default values for these parameters, can be overwritten
"""



"""
Wavelet Parameters

"""

# number of wavelets (wavelet frequencies) to use
numPeriods = 25

# dimensionless Morlet wavelet parameter
omega0 = 5

# minimum frequency (longest period)
minF = 1

# maximum frequency (shortest period), set to <= nyquist frequency
maxF = 30



"""
T-SNE Parameters

"""


# perplexity (2^H where H is the transition entropy)
perplexity = 32

# relative convergence for t-sne
relTol = 0.0001

# Number of output dimensions for t-sne
num_tsne_dims = 2

# binary search tolerance for finding pointwise transitions
sigmaTol = 0.00001

# maximum number of non-zero neighbours in P
maxNeighbours = 200

# initial momentum
momentum = 0.5

# momentum after change
final_momentum = 0.8

# momentum switch iteration
mom_switch_iter = 250

# iteration at which dummy P-values stop
stop_lying_iter = 125

# degree of p-value expansion for early iterations
lie_multiplier = 4

# maximum number of t-sne iterations
max_iter = 1000

# initial learning rate
epsilon = 500

# minimum gain for delta-bar-delta
min_gain = 0.1

# readout for t-sne
tsne_readout = 1

# embedding batchsize
embedding_batchsize = 20000

# max iterations for Nelder-Mead algo
maxOptimIter = 100

# number of points in training set (if training set is being used)
trainingSetSize = 35000

# local neighbourhood for training set
kdNeighbours = 5

# t-sne training set stop
training_relTol = 0.002

# t-sne training set perplexity
training_perplexity = 20

# number of points from each training set file (if using multiple files)
training_numPoints = 10000

# Minimum training set template length
minTemplateLength = 1



