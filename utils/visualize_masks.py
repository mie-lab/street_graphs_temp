# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 03:30:36 2019

@author: henry
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

cities = ['Berlin','Istanbul','Moscow']

mask_dict = pickle.load(open( os.path.join('utils', 'masks.dict'), "rb" ) )


plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (16,9)

fontSizeAll = 20
font = {     'size'   : fontSizeAll}
matplotlib.rc('font', **font)

# plot 3 tiles per city
for city in cities:
    
    mask_ = mask_dict[city]['mask']
    sum_ = mask_dict[city]['sum']
    # mask_ =  mask_dict[city]['sum'] > 0
    
    fig,ax = plt.subplots(1,3, sharey=True)
    ax[0].spy(mask_)
    ax[0].set_title('Binary mask')
    
    im = ax[1].imshow(sum_)
    ax[1].set_title('Image of sum')
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
#    plt.colorbar(im, ax=ax[1])
    #plt.colorbar()
    
    im = ax[2].imshow(np.log10(sum_+1))
    ax[2].set_title('Image of log sum')
    
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    
#    plt.colorbar(im, ax=ax[2])
    #plt.colorbar()
    
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        
    plt.tight_layout
    plt.savefig(os.path.join('utils','images', city+'_masks.png'), 
                             bbox_inches='tight')
    plt.savefig(os.path.join('utils','images', city+'_masks.pdf'),
                bbox_inches='tight')

# plot logsum for all cities
fig,ax = plt.subplots(1,3, sharey=True)
for city_ix, city in enumerate(cities):
    
    mask_ = mask_dict[city]['mask']
    sum_ = mask_dict[city]['sum']
    
   
   
    im = ax[city_ix].imshow(np.log10(sum_+1))
    ax[city_ix].set_title(city)
    
    divider = make_axes_locatable(ax[city_ix])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    
#    plt.colorbar(im, ax=ax[city_ix])
    #plt.colorbar()
    
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        
    plt.tight_layout
    plt.savefig(os.path.join('utils','images', 'logsum_allcities.png'), 
                             bbox_inches='tight')
    plt.savefig(os.path.join('utils','images', 'logsum_allcities.pdf'), 
                             bbox_inches='tight')



import skimage
import skimage.feature
import skimage.viewer
import sys

data = np.log10(sum_+1)
data = np.log10(sum_+1)/np.max(np.log10(sum_+1)) * 255
data = sum_+1

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
highpass_3x3 = ndimage.convolve(data, kernel)
plt.imshow(highpass_3x3>0)
plt.imshow(np.log10(highpass_3x3))

# A slightly "wider", but sill very simple highpass filter 
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass_5x5 = ndimage.convolve(data, kernel)
plt.imshow(highpass_5x5>0)
plt.imshow(np.log10(highpass_5x5)>0)

lowpass = ndimage.gaussian_filter(data, 5)
gauss_highpass = data - lowpass
plt.imshow(gauss_highpass)
plt.imshow(np.log10(gauss_highpass)>0)



plt.imshow(data)

# def plot_cdf(ax, x, label, n_bins=100):
#     # plot the cumulative histogram
#     n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
#                                cumulative=True, label=label)






# #------------------- hist of 

# fig, ax = plt.subplots(figsize=(8, 4))

# #fig, axs = plt.subplots(1,3)
# for ix, city in enumerate(cities):
    
#     mask_ = mask_dict[city]['mask']
#     sum_ = mask_dict[city]['sum']
    
#     sum_ = np.log10(sum_+1)
    
#     plot_cdf(ax, x=sum_.ravel(), label=city, n_bins=500)
    


# # tidy up the figure
# ax.grid(True)
# ax.legend(loc='right')
# ax.set_title('Cumulative step histograms')
# ax.set_xlabel('Log sum over all channels (mm)')
# ax.set_ylabel('Likelihood of occurrence')
# plt.tight_layout()

# plt.savefig(os.path.join('utils','images','hist_of_sum.png'))

