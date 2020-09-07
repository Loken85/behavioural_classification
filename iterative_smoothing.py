# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:28:49 2019

@author: proxy_loken

Smoothing functions for pre-processing automatic annotation (smooths deeplabcut
output to pass to behavioural classification)


"""

import pandas
import numpy as np


import scipy.io as sio


# should be passed a pandas dataframe of labelled features (x,y,likelihood)
# top level should be the feature names (eg. 'N1','SG1' etc.)
def iter_smooth(data,threshold = 0.15):
    sdata = data.copy()
    for feature in list(sdata.columns.levels[0]):
        #fdata = sdata[feature]
        # Don't do it this way. Itterrows does not run in order, and has terrible
        # run time (why pandas, why?)
#        for i, row in sdata[feature].iterrows():
#            if row['likelihood'] < threshold:
#                print ('ind: ' + str(i))
#                print('lh: ' + str(row['likelihood']))
#                gap = calc_gap(i,sdata[feature]['likelihood'],threshold)
#                print('gap: ' + str(gap))
#                g_vals = gen_curve(sdata[feature][i-1:i+gap+1])
#                #sdata[feature].iloc[i:i+gap-1] = g_vals
#                sdata.loc[i:i+gap,feature] = g_vals
                
        g_inds = sdata[feature]['likelihood'].index[sdata[feature]['likelihood']< threshold].tolist()
        ind = g_inds[0]
        curr_ind = 0
        while ind <= g_inds[-1]:
            print('index: ' +str(ind))
            gap = calc_gap(ind,sdata[feature]['likelihood'],threshold)
            print('gap: ' +str(gap))
            if ind > 0:
                g_vals = gen_curve(sdata[feature][ind-1:ind+gap+1])
            else:
                g_vals = gen_curve(sdata[feature][ind:ind+gap+1])
                
            print('curve size: ' + str(len(g_vals)))
            # multi-slice indexing of ranges (as above) is exclusive...
            # multi-index slice ranges (as below) are inclusive (because who knows)
            sdata.loc[ind:ind + len(g_vals)-1, feature] = g_vals.values
            
            curr_ind += gap
            if curr_ind < (len(g_inds)-1):
                ind = g_inds[curr_ind]
            else:
                ind = g_inds[-1] + 1
        
    
                
    
    return sdata


# calculate the length of the low confidence gap. A single low confidence value
# has a gap length of one
def calc_gap(ind, likelihoods,threshold):
    gap_length = 0
    val = likelihoods[ind]
    while val < threshold:
        gap_length += 1
        ind += 1
        if ind >= len(likelihoods):
            gap_length -+ 1
            break;
        else:
            val = likelihoods[ind]
    
    
    
    return gap_length


# function to generate the new curve between the first and last indices, which should be true predictions
# currently a linear smoother, can change this in future if needed
def gen_curve(c_data):
    
    g_vals = c_data.copy()
    
    delta_x = -(c_data['x'].iloc[0]-c_data['x'].iloc[-1]) / (len(c_data['x']))
    
    delta_y = -(c_data['y'].iloc[0]-c_data['y'].iloc[-1]) / (len(c_data['y']))
    # counts progress across the gap
    ct = 0
    
    for j, row in c_data.iterrows():
        g_vals.loc[j,'x'] = c_data['x'].iloc[0] + (ct*delta_x)
        g_vals.loc[j,'y'] = c_data['y'].iloc[0] + (ct*delta_y)
        g_vals.loc[j,'likelihood'] = 1
        ct += 1
    
    # trim the first and last values to return only the generated curve
    g_vals = g_vals.iloc[1:-1]
    
    return g_vals






