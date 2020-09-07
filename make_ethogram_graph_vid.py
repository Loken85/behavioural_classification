#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:19:21 2019

@author: seamans

Functions for creating videos with embedded ethogram and behavioural space graph
( requires deeplabcut package and configuration file currently (TODO: remove this dependency))


"""



import os.path
#import sys
import argparse, os

import numpy as np
from tqdm import tqdm
from pathlib import Path

import pickle


import platform
import scipy as sc
import cv2

import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
elif platform.system() == 'Darwin':
    mpl.use('WxAgg') #TkAgg
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from skimage.util import img_as_ubyte
from skimage.draw import rectangle
from deeplabcut.utils.video_processor import VideoProcessorCV as vp # used to CreateVideo

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def make_ethogram_video(clip,t_out, labels, l_frames, xi, yi):
    
    colormap = get_cmap((np.max(labels)+1))
    colorclass=plt.cm.ScalarMappable(cmap=colormap)
    C=colorclass.to_rgba(np.linspace(0,1,(np.max(labels)+1)))
    colors=(C[:,:3]*255).astype(np.uint8)
    ny, nx = clip.height(), clip.width()
    #ny, nx = clip.height(), clip.width()
    fps = clip.fps()
    nframes = len(t_out)
    duration = nframes/fps
    
    print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
    print("Overall # of frames: ", nframes, "with frame dimensions: ",nx,ny)
         
    for index in tqdm(range(nframes)):
        image = clip.load_frame()
        beh_value = l_frames[index]
        
        rr, cc = rectangle((5,5),end=(25,25))
        image[rr, cc,:] = colors[np.int(beh_value)]
        # Add numeric value of behaviour to video
        image = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,('' + str(beh_value) ),(35,25),font,1,(0,0,0),1,cv2.LINE_AA)
        
        
        # plot the density space and add label for current frame
        fig = Figure(figsize=(3,3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        
        ax = fig.add_subplot(111)
        
        ax.pcolormesh(xi, yi, labels)
        
        x,y = t_out[index]
        ax.scatter(x,y,s=10,c='red',marker='+')
        
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        graph = np.frombuffer(s, np.uint8).reshape((height,width,4))
        graph = graph[:,:,0:3]
        
        
        image[-300:,-300:,:] = graph
        #h1, w1 = image.shape[:2]
        #h2, w2 = graph.shape[:2]

        #create empty matrix
        #vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        #combine 2 images
        #vis[:h1, :w1,:3] = image
        #vis[:h2, w1:w1+w2,:3] = graph
        
        #frame = np.copy(vis)
        frame = np.copy(image)
        clip.save_frame(frame)
        
    clip.close()
    
    
def create_ethogram_video(data, video, videotype='mp4',codec='mp4v'):
    
    
    vname = str(Path(video).stem)
    videooutname=os.path.join(vname + '_ethogram_graph.mp4') 
    
    clip = vp(video,sname = videooutname, codec=codec)
    
    with open(data, 'rb') as fp:
        l_data = pickle.load(fp)
    
    t_out = l_data['t_out']
    labels = l_data['labels']
    l_frames = l_data['l_frames']
    xi = l_data['xi']
    yi = l_data['yi']
    
    
    make_ethogram_video(clip,t_out,labels, l_frames, xi, yi)




# Variables for testing

data = '/home/seamans/DLC/data.p'
video = '/home/seamans/DLC/ethogram_testing/example_stack_copy.mp4'

create_ethogram_video(data,video)





#if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('data')
    #parser.add_argument('video')
    #cli_args = parser.parse_args()
    
    
    
    




