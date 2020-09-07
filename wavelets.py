# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:48:08 2019

@author: proxy_loken



Functions and parameters for wavelet space embedding

"""
import pywt
import numpy as np
import pandas as pd


# calls the continuous wavelet transform using the morlet wavelet on the data
# vector at the provided scales (scales relate to the sampling frequency 'sfreq'
# with scale 1 = sampling frequency
def morlet_cwt(data, scales, sfreq, width=8):
    # pad with zero to flatten edge artifacts
    data = zero_pad(data, scales[0], width)
    scalogram, freqs = pywt.cwt(data, scales, 'morl', sfreq)
    #trim zeros
    p_width = np.ceil(scales[0]*width)
    p_width = p_width.astype(int)
    scalogram = scalogram[:,p_width:-p_width]
    return scalogram, freqs

# as above, but using the complex morlet with additional parameters for 
# bandwidth (concentration of energy in the frequency domain) and centre
# frequency (the constant of proportionality between scale and frequency)
def cmorlet_cwt(data, scales, sfreq, width=8, bandwidth=1.5, c_freq = 1):
    #pad with zeros
    data = zero_pad(data, scales[0], width)
    wavelet = 'cmor'+str(bandwidth)+'-'+str(c_freq)
    scalogram, freqs = pywt.cwt(data, scales, wavelet, sfreq)
    #trim zeros
    p_width = np.ceil(scales[0]*width)
    p_width = p_width.astype(int)
    scalogram = scalogram[:,p_width:-p_width]
    # extract the real component from the scalogram (currently discards the
    # imaginary component, keep if you need phase info in future)
    r_scalogram = np.real(scalogram)
    # take the absolute value of the amplitudes
    r_scalogram = np.abs(r_scalogram)
    
    return r_scalogram, freqs

# calculate scales from the number of periods
def calculate_scales(minf, maxf, sfreq, numPeriods):
    
    srate = 1/sfreq
    # linear frequency set. Emphasizes higher freqs. Scale using log to emphasize
    # low end
    freqs = np.linspace(minf,maxf, numPeriods)
    scales = 1/(np.multiply(freqs,srate))
    
    return scales, freqs


# performs the wavelet transform on the provided data set. Returns a dataframe of the transformed data and a numpy array
# with length(scales) * size(data,1) rows, of size(data,0)
def wavelet_transform(data, scales, freqs, sfreq):
    
    # create index for new dataframe to hold transformed data
    bparts = pd.unique(data.T.index.get_level_values(0))
    coords = pd.unique(data.T.index.get_level_values(1))
    mindex = pd.MultiIndex.from_product([bparts,coords,freqs],names=['bodyparts','coords','freqs'])
    # number of wavelets
    fs = len(freqs)
    # length of data
    dlen = len(data.loc[:,bparts[0]])
    cols = len(bparts)*len(coords)*fs
    
    
    #empty array to hold arrays of amplitudes
    scalos = np.zeros((cols, dlen))
    
    ind = 0
    for feature in list(bparts):
        for coord in list(coords):
            # this is where the wavelet function is defined. change here if
            # necessary
            # TODO: make wavelet function an argurment to this function
            # non-complex morlet:
            #scalo,freq = morlet_cwt(data[feature][coord], scales, sfreq)
            # complex morlet:
            scalo,freq = cmorlet_cwt(data[feature][coord], scales, sfreq)
            scalos[ind*fs:ind*fs+fs,:] = scalo
            ind = ind +1
            
        
    # slice array into dataframe labelled by bodypart, coord, freq
    scalogram = pd.DataFrame(np.transpose(scalos),index=range(dlen),columns=mindex)
    
   
    return scalogram, scalos


# performs the wavelet transform on the provided data set (as a numpy array). Returns a numpy array
# with length(scales) * size(data,1) rows, of size(data,0)
def wavelet_transform_np(data, scales, freqs, sfreq):
    
    
    # number of wavelets
    fs = len(freqs)
    # length of data
    dlen = np.size(data,0)
    cols = np.size(data,1)*fs
    
    
    #empty array to hold arrays of amplitudes
    scalos = np.zeros((cols, dlen))
    
    ind = 0
    for feature in range(np.size(data,1)):
        for coord in range(1):
            # this is where the wavelet function is defined. change here if
            # necessary
            # TODO: make wavelet function an argurment to this function
            # non-complex morlet:
            #scalo,freq = morlet_cwt(data[feature][coord], scales, sfreq)
            # complex morlet:
            scalo,freq = cmorlet_cwt(data[:,ind], scales, sfreq)
            scalos[ind*fs:ind*fs+fs,:] = scalo
            ind = ind +1
            
        
        
   
    return scalos

# Pads a vector with zeros as needed for the wavelet transform
def zero_pad(data,scale,width):
    #default width for morlet wavelet
    p_width = np.ceil(scale*width)
    p_width = p_width.astype(int)
    pdata = np.pad(data, p_width, 'constant')
    
    #TODO
    return pdata
