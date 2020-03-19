import time
import os
import random

is_commitserver = os.path.isfile('on.commitserver')
is_4gpuserver = os.path.isfile('on.4gpuserver')
is_domlocal = os.path.isfile('on.domlocal')
is_henlocal = os.path.isfile('on.henlocal')
config = dict()


# data set configuration
config['dataset'] = {}

# Where raw data are stored.
# config['dataset']['source_root'] = r'D:\Codes\Data'
# config['dataset']['target_root'] = r'data'
if is_commitserver:
    print('is_commitserver')
    config['dataset']['source_root'] = '/data/traffic4cast/data_raw'
    config['dataset']['target_root'] = '/data/traffic4cast/data'
elif is_4gpuserver:
    print('is_4gpuserver')
    config['dataset']['source_root'] = '/home/henry/data_raw'
    config['dataset']['target_root'] = '/home/henry/data'
elif is_domlocal:
    print('is_domlocal')
    config['dataset']['source_root'] = r'C:\Data\Data\here_challenge'
    config['dataset']['target_root'] = r'C:\Data\Data\here_challenge\preprocessed'
elif is_henlocal:
    print('is_henlocal')
    config['dataset']['source_root'] = r'C:\Data\traffic4cast'
    config['dataset']['target_root'] = r'data\data'   
else:
    print('else')
    config['dataset']['source_root'] = r'C:\Data\traffic4cast'
    config['dataset']['target_root'] = r'data\data'

config['dataset']['return_features'] = True
config['dataset']['cities'] = ['Moscow']
##################################################################



# model statistics 
config['model'] = {}

config['add_coords'] = True
if config['add_coords']:
    config['model']['in_channels'] = 38
else:
    config['model']['in_channels'] = 36


config['model']['depth'] = 1

# Kipfnet
config['model']['KIPF'] = {}
config['model']['KIPF']['nh1'] = 64
config['model']['KIPF']['K'] = 7

# Kipfnet2
config['model']['KIPF2'] = {}
config['model']['KIPF2']['nh1'] = 56
config['model']['KIPF2']['nh2'] = 24
config['model']['KIPF2']['K'] = [12,8,4]

# KipfNet_res2
config['model']['KipfNet_res2'] = {}
config['model']['KipfNet_res2']['nh1_1'] = 64
config['model']['KipfNet_res2']['nh1_2'] = 64
config['model']['KipfNet_res2']['K'] = 7

# KipfNet_res3
config['model']['KipfNet_res3'] = {}
config['model']['KipfNet_res3']['nh1_1'] = 64
config['model']['KipfNet_res3']['nh1_2'] = 64
config['model']['KipfNet_res3']['nh1_3'] = 64
config['model']['KipfNet_res3']['K'] = 7

# GUNET
config['model']['GUNET'] = {}
config['model']['GUNET']['pool_ratios']= [0.10, 0.50]
config['model']['GUNET']['depth']= 3
config['model']['GUNET']['nh1']= 64


config['cont_model_path'] = None  # Use this to continue training a previously started model.

# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = True
if is_commitserver or is_4gpuserver:
    config['dataloader']['num_workers'] = 16
    config['dataloader']['batch_size'] = 4
elif is_henlocal:
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 2
else:
    config['dataloader']['num_workers'] = 4
    config['dataloader']['batch_size'] = 2

#Graph creation
# I don't have any idea of an optimal value for the mask threshold. In the past I often used 50'000
# I noticed that 200'000 is probably too much. 
config['mask_threshold'] = 10


# optimizer
config['optimizer_name'] = 'SGD'
config['optimizer'] = {}
config['optimizer']['lr'] = 0.001
# config['optimizer']['weight_decay'] =0.00002
config['optimizer']['momentum'] = 0.9
config['optimizer']['nesterov'] = True

# lr schedule
config['lr_step_size'] = 4
config['lr_gamma'] = 0.1

# early stopping
config['patience'] = 10
config['num_epochs'] = 5
config['print_every_step'] = 10

if is_commitserver or is_4gpuserver:
    config['device_num'] = 1
    config['debug']= False
elif is_domlocal:
    config['device_num'] = 'cpu'
    config['debug']= True
else:
    config['device_num'] = 0
    config['debug']= True

config['model_name'] = 'KIPF'