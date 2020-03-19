# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:19:13 2019

@author: henry
"""

import torch
import numpy as np
import time
import sys, os
sys.path.append(os.getcwd())
from videoloader import trafic4cast_dataset, test_dataloader
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    source_root = r'C:\Data\traffic4cast'
    target_root = r'data\data'
    plot_flag  = False
    
    mask_dict = {}
    mask_dict_target = {}
    threshold = 0
    log_dict = {}

    for city in ['Berlin','Moscow', 'Istanbul']:
        
        overall_sum = 0
        overall_sum_target = 0
        overall_sum_channel = 0
        logging = []
        
        kwds_dataset = {'cities': [city], 'filter_test_times': True}
        kwds_loader = {'shuffle': False, 'num_workers':12,
                'batch_size': 25}
    
        dataset_train = trafic4cast_dataset(source_root, target_root,
                                            split_type='training', **kwds_dataset)
        dataset_val = trafic4cast_dataset(source_root, target_root,
                                          split_type='validation', **kwds_dataset)
#        dataset_test = trafic4cast_dataset(source_root, target_root,
#                                          split_type='test', **kwds_dataset)
        
        train_loader = torch.utils.data.DataLoader(dataset_train, **kwds_loader)
        val_loader = torch.utils.data.DataLoader(dataset_val, **kwds_loader)
#        test_loader = torch.utils.data.DataLoader(dataset_test, **kwds_loader)
        
#        loader_list = [train_loader, val_loader, test_loader]
        loader_list = [train_loader, val_loader]
        # sum up all available data for a city
        
        for loader in loader_list:
            if plot_flag:
                fig, axs = plt.subplots(1,2, figsize=(24,12))

            for batch_idx, (data, Y, context) in enumerate(loader):
                
                
                batch_sum = torch.sum(data, (0,1,2))
                overall_sum = overall_sum + batch_sum 
                
                batch_sum_channel = torch.sum(data, (0,1,2))
                overall_sum_channel = overall_sum_channel + batch_sum 
                
                batch_sum_target = torch.sum(Y, (0,1,2))
                overall_sum_target = overall_sum_target + batch_sum_target 

                if (batch_idx+1) % 1 == 0:

                    batch_mask = (batch_sum  > threshold)
                    overall_mask = (overall_sum  > threshold)
                    
                    nonzeros = np.sum(overall_mask.numpy())/overall_mask.numel() * 100
                    logging.append((batch_idx,nonzeros))
                    
                    print('{}, {} [{}/{}] {:.2f}% non-zeros'.format(
                            city, loader.dataset.split_type,
                            batch_idx * len(data), len(loader.dataset),
                            nonzeros))
                    if plot_flag:   
                        axs[0].spy(batch_mask)
                        axs[1].spy(overall_mask)
                        plt.pause(0.01)
                    
        
                    
                
                
        overall_sum_channel = overall_sum_channel.numpy()
        overall_sum = overall_sum.numpy()
        overall_mask = (overall_sum > threshold)
        overall_sum_target = overall_sum_target.numpy()
        overall_mask_target = (overall_sum_target > threshold)        

        mask_dict[city] = {'mask': overall_mask, 'sum': overall_sum, 'channel_sum': overall_sum_channel}
        mask_dict_target[city] = {'mask': overall_mask_target, 'sum': overall_sum_target}

        print(np.sum(overall_mask)/overall_mask.size,"% non-zeros in ", city)
        
    print('writing file')
    pickle.dump( mask_dict, open( os.path.join('.',"utils","masks_testtimes.dict"), "wb" ) )
    pickle.dump( mask_dict_target, open( os.path.join('.',"utils","masks_target_testtimes.dict"), "wb" ) )
    print('done')
