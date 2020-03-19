import torch

import numpy as np
import time

import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'baselines'))
from videoloader import trafic4cast_dataset, test_dataloader
from henry.config import config
import matplotlib.pyplot as plt
from utils.create_submission import submission_writer
from baselines import persistence_forecast
from example_submission import predict_submit
import numpy as np
import os, csv
import sys, getopt
import h5py
from utils.create_submission import submission_writer


cities = ['Berlin']
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174,222, 258]

def list_filenames(directory):
    filenames = os.listdir(directory)
    return filenames

def load_test_file(file_path):
    """
    Given a file path, loads test file (in h5 format).
    Returns: tensor of shape (number_of_test_cases = 5, 3, 3, 496, 435) 
    """
    # load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant test cases
    data = [data[0:]]
    data = np.stack(data,axis=0)
    # rescale
    data = data.astype(np.float32)
    return data

def load_input_data(file_path, indicies):
    """
    Given a file path, load the relevant training data pieces into a tensor that is returned.
    Return: tensor of shape (number_of_test_cases_per_file =5, 3, 495, 436, 3)
    """
    # load h5 file into memory.
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant training data pieces
    data = [data[y:y+3] for y in indicies]
    data = np.stack(data, axis=0)

    # type casting
    data = data.astype(np.float32)
    return data


def work_out_score(submission_path, golden_path, output_path, output_prefix, split_type='test'):
    prefix = output_prefix
    """
    Work out the relevant mse score for a given submitted unpacked test file and having the 
    path to the golden file as well.
    Assumptions: 
        - submitted file directory structure is 
            city/city_test
          for city = Berlin, Istanbul and Moscow.
        - Moreover, it assumes that the file names within the submitted directory are the same as
          in the golden file (and same as in the public file given out).
        - Assumes file types are the same as public and golden data, i.e. h5 formats (that python will read in and 
          convert to np arrays.
    Output: writes relevant files and returns overall mse value (which might be superfluous but just in case).

    """
    
    # create necessary writers:
    log_file_path = os.path.join(output_path, prefix+'.log')
    log_file = open(log_file_path, 'w')
    log_writer = csv.writer(log_file, lineterminator="\n")
    score_file_path = os.path.join(output_path, prefix+'.score')
    score_file = open(score_file_path, 'w')
    score_writer = csv.writer(score_file, lineterminator="\n")
    extended_score_file_path = os.path.join(output_path, prefix+'.extended_score')
    extended_score_file = open(extended_score_file_path, 'w')
    extended_score_writer = csv.writer(extended_score_file, lineterminator="\n")
    
    # iterate through cities.
    overall_mse = 0.0
    for city in cities:
        city_mse = 0.0
        # get file names
        data_dir_golden = os.path.join(golden_path, city, city+'_'+split_type)
        data_dir_sub = os.path.join(submission_path, city, city+'_'+split_type)
        # we assume these are now the same as the golden data set file names in the relevant directory.
        sub_files = list_filenames(data_dir_sub)
        # iterate through assumed common file names, load the data and determine the MSE and store and iterate.
        filecount = 0.0
        for f in sub_files:
            filecount = filecount + 1.0
            data_sub = load_test_file(os.path.join(data_dir_sub,f))
            
            if split_type == 'test':
                data_golden = load_test_file(os.path.join(data_dir_golden,f))
            else:
                indicies = utcPlus3
                if city == 'Berlin':
                    indicies = utcPlus2
                indicies = np.asarray(indicies)
                data_golden = load_input_data(os.path.join(data_dir_golden,f), indicies)

            # just for debugging purposes
            # print(data_sub.shape)
            # print(data_golden.shape)
            # calculate MSE
            mse = (np.square(np.subtract(data_sub[...,0],data_golden[...,0]))).mean(axis=None)
            mse2 = (np.square(np.subtract(data_sub[0,...,1],data_golden[0,...,1]))).mean(axis=None)
            mse3 = (np.square(np.subtract(data_sub[0,...,2],data_golden[0,...,2]))).mean(axis=None)
            # now book keeping.
            city_mse += mse
            log_writer.writerow([city, f, mse])
            log_file.flush()
            print("City: {} - File: {}  --- > {}".format(city, f, mse))
            print("City: {} - File: {}  --- > {}".format(city, f, mse2))
            print("City: {} - File: {}  --- > {}".format(city, f, mse3))
        city_mse /= filecount
        overall_mse += city_mse/3.0
        print(city, city_mse)
        extended_score_writer.writerow([city, city_mse])
        extended_score_file.flush()

    score_writer.writerow([overall_mse])
    score_file.flush()
    # closing all files
    score_file.close()
    extended_score_file.close()
    log_file.close()

    return overall_mse

    

if __name__ == "__main__":
    
    # usage for debugging:
    # - copy the raw validation dataset of Berlin into 
    #   a new folder [source_root]
    
    source_root = r"C:\Data\traffic4cast2" # raw data
    target_root = "data2" # preprocessed raw data
    submission_root = os.path.join("submission", "validation2") # submission file 
    golden_path=r"C:\Data\traffic4cast2" # ground truth path for evaluation
    
    reduce = True

    kwds_dataset = {'source_root':source_root,
                    'target_root': target_root,
                    'reduce': reduce,
                    'split_type': 'validation',
                    'filter_test_times': True}

    # it is not yet clear to me if more
    # workers speed up the prediction...
    kwds_loader = {'shuffle': False,
                   'num_workers':0,
                   'batch_size': 5
                   }

    writer_kwds={'init_dir':True}
    
    # create dataset
    dataset = trafic4cast_dataset(**kwds_dataset)
    loader = torch.utils.data.DataLoader(dataset, **kwds_loader)
    
    (data_loaded, target_loaded) = dataset[0]
    

    print(dataset.target_root)

    predict_submit(loader, submission_root=submission_root, print_interval=print_interval, 
                  reduce=reduce, do_prediction=False, writer_kwds=writer_kwds)
    print('submission finished')
    
    rest = work_out_score(submission_path=submission_root, golden_path=golden_path,
            output_path='utils', output_prefix='val_', split_type='validation')

    print(rest)



    print('done')
            
# test reduced submission
sm = submission_writer(loader.dataset, submission_root=submission_root, **writer_kwds)


def compare_by_dimension_red(dataset, gt_data,i):
#    k = i*15
    (data_loaded, target_loaded)= dataset[i]
    data_all = np.concatenate((data_loaded.numpy(),target_loaded.numpy()),axis=0)
    e = 0
    for t in range(15):
        for c in range(3):
             j = t*3 +c
             k = t*3
             e1 = np.sum(np.abs(data_all[j,...] - gt_data[i+t,:,:,c]))
        
    
    writer_dict = sm.fake_submission_writing(target_loaded, i)
    ii = utcPlus2[i] 
    d_sub = writer_dict['submission_data']
    e2 = np.sum(np.abs(d_sub - gt_data[i+ii:i+3+ii,:,:,:]))
    print(e2)





for i in range(287-15):
    compare_by_dimension_red(dataset, gt_data,i)
    
    

def compare_by_dimension(dataset, gt_data,i):
#    k = i*15
    (data_loaded, target_loaded)= dataset[i]
    data_all = np.concatenate((data_loaded.numpy(),target_loaded.numpy()),axis=0)
    for t in range(15):
        for c in range(3):
             print(np.sum(np.abs(data_all[t, c,...] - gt_data[i+t,:,:,c])))


for i in range(287-15):
    compare_by_dimension(dataset, gt_data,i)
    

    
    
    
