"""Summary

Attributes:
    cities (list): Description
"""

import torch

import numpy as np
import time
import numpy as np
import os, csv
import sys, getopt
import h5py
import shutil
from pathlib import Path



import sys, os
sys.path.append(os.getcwd())
from videoloader import trafic4cast_dataset, test_dataloader


def read_hdf5_file(filename):

    with h5py.File(filename, 'r') as f:
        dset = f['array']
        data = dset[:]
        f.close()

    return data

class submission_writer():

    """Object that writes predictions in traffic4cast submission file structure
    
    Attributes:
        ds (TYPE): Traffic4cast Dataset object
        init_dir (Bool, optional): If True it will delete the given submission folder
            and initialize it with zero filled .hdf5 files 
        submission_root (TYPE): Directory to create the submission files
    """
    
    def __init__(self, ds, batch_size, submission_root=os.path.join('submission', 'test'),
             init_dir=True, cities = ['Berlin', 'Istanbul', 'Moscow']):
        """Store input as attributes and initialize file system
        
        Args:
             ds (TYPE): Traffic4cast Dataset object
            submission_root (TYPE): Directory to create the submission files
            init_dir (Bool, optional): If True it will delete the given submission folder
            and initialize it with zero filled .hdf5 files 
        """
        self.ds = ds
        self.submission_root = submission_root
        self.init_dir = init_dir
        self.batch_size = batch_size
        self.cities = cities
        
        
        
        if self.init_dir and os.path.exists(self.submission_root):
            shutil.rmtree(self.submission_root, ignore_errors=True)
        if self.init_dir:
            self._init_output_files(self.ds.target_root, self.submission_root, self.ds.split_type)
    
    def _get_orig_ix(self, ix, batch_size):
        """Returns the minimum and maximum index of a submission batch
        
        Args:
            ix (TYPE): Index as given by the data loader
            batch_size (TYPE): batch size as given by the data loader
        
        Returns:
            Numpy Array: Array of the indicies within a batch 
        """
        min_ix = batch_size*ix
        max_ix = batch_size*(ix+1)
        
        return np.arange(min_ix, max_ix)
        
    def _get_timestamps(self, ix_list, ix, orig_ixs):
        """Returns the first and the last position in a list of an index.
        
        Args:
            ix_list (List): A sorted list of indicies, indicies appear multiple times
            ix (Int): Index
        
        Returns:
            Tuples: Returns two tuples.
        """
        first_ix  = ix_list.index(ix) 
        last_ix = (len(ix_list) - 1 - ix_list[::-1].index(ix))
        
        # position of relevant data in array
        arraystamps = (first_ix, last_ix +1)
        # position of relevant data in submission file
        filestamps = (orig_ixs[first_ix]%5, orig_ixs[last_ix]%5 +1)
        
        # add 1 at the end because they are meant as list indices
        return arraystamps, filestamps 
        
        
    
    def _list_filenames(self, directory):
        """List all filenames in a directory
        From: https://github.com/iarai/NeurIPS2019-traffic4cast
        Args:
            directory (string): directory to list files
        
        Returns:
            TYPE: list
            List of filenames
        """
        filenames = os.listdir(directory)
        return filenames
    
    def _get_city(self, path, root):
        """Extract the city name of standard traffic4cast like data structure path
        
        Args:
            path (str): Traffic4cast path: [root + city + city_test + filename]
            root (str): Root directory
        
        Returns:
            TYPE: Description
        """
        city_path = Path(path.replace(root,''))
        city_name = city_path.parts[1]
        return city_name
        
    def _write_full_file(self, data, filename):
        """writes data into a .hdf5 file
       From: https://github.com/iarai/NeurIPS2019-traffic4cast 
        Args:
            data (Array):
            filename (string): 
        """
        with h5py.File(filename, 'w', libver='latest') as f:
            dset = f.create_dataset('array', shape=(data.shape),
                                    data = data,
                                    compression='gzip', 
                                    compression_opts=9
                                    )
            f.close()
        
    def _create_directory_structure(self, root, split_type):
        """Creates traffic4cast like directory structure 
        From: https://github.com/iarai/NeurIPS2019-traffic4cast
        Args:
            root (str): Root directory where the file structure should be created 
        """
        city_init_list = []
        if 'Berlin' in self.cities:
            berlin = os.path.join(root, "Berlin","Berlin_" + split_type)
            city_init_list.append(berlin)
        if 'Istanbul' in self.cities:
            istanbul = os.path.join(root, "Istanbul","Istanbul_" + split_type )
            city_init_list.append(istanbul)
        if 'Moscow' in self.cities:
            moscow = os.path.join(root, "Moscow", "Moscow_" + split_type)
            city_init_list.append(moscow)
        
        for path in city_init_list:
            try:
                os.makedirs(path)
            except OSError:
                print("Warning: Path {} already exists".format(path))
                
    def _init_output_files(self, input_path, output_path, split_type):
        """
        write outdata into each submission folder structure at out_path, cloning
        filenames in corresponding folder structure at input_path.

        Inspired from: https://github.com/iarai/NeurIPS2019-traffic4cast
       
        Args:
            input_path (TYPE): Description
            output_path (TYPE): Description
        """
        self._create_directory_structure(output_path, split_type)
        out_data = np.zeros((5,3,495,436,3))
        for city in self.cities:
            # set relevant list
            data_dir = os.path.join(input_path, city, city+'_'+split_type)
            sub_files = self._list_filenames(data_dir)
            for f in sub_files:
                # load data
                outfile = os.path.join(output_path, city, city+'_'+split_type,f)
                self._write_full_file(out_data, outfile)

        
    def _write_part_to_file(self, submission_data, submission_filename, filestamps):
        """Write data into an initialized .hdf5 file
        Data could cover only parts of the file.

        Args:
            submission_data (Numpy array): Shape (-1, 3, 495, 436, 3)
            submission_filename (TYPE): Description
            tstamps (TYPE): Description
        """
        with h5py.File(submission_filename, 'a') as f:
            dset = f['array']
            dset[filestamps[0]:filestamps[1], ...] = submission_data
            f.close()
        
    def _restore_file_order(self, submission_data):
        """Transforms data into the original shape for submission and to numpy
        Shape input:  [-1, 3, 3, 495, 436]
        Shape output: [-1, 3, 495, 436, 3]

        Args:
            submission_data (Torch array):
        
        Returns:
            Numpy arary:  
        """
        submission_data = submission_data.permute(0,1,3,4,2)
        return submission_data.numpy()
    
    def _restore_reduced_file_order(self, submission_data_red, batch_size):
        """Transforms data from its reduced from into the original shape for submission and to numpy
        
        Shape input: [-1, 9, 495, 436]
        Shape output: [-1, 3, 495, 436, 3]

        Args:
            submission_data_red (Torch array): Description
            batch_size (TYPE): batch size from data loader
        
        Returns:
            TYPE: Description
        """
        # move axis to unstack in the right order
        data_temp = np.moveaxis(submission_data_red.numpy(), 1, 3)
        
        #unstack and move axis back
        data_temp = np.reshape(data_temp,(batch_size,495,436,3,3))
        submission_data = np.moveaxis(data_temp,3,1)
        
        return submission_data        
    
    def _ensure_batchsize_dimension(self, submission_data):
        """Ensures that the batchsize has its proper dimension.
        
        All transformations expect a proper dimension for the batch size 
        even for batch_size = 1
        
        Dimensions should be:
            reduce == True: (batch_size, 9,)
        
        Args:
            submission_data (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        
        shp = submission_data.shape
      
        if self.ds.reduce and len(shp) == 3:
                return submission_data.reshape(1, shp[0], shp[1], shp[2])
        elif  not self.ds.reduce and len(shp) == 4:
                return submission_data.reshape(1, shp[0], shp[1],
                                               shp[2], shp[3])
        else:
            return submission_data
        
        
    def write_submission_data(self, submission_data, idx):
        """Summary
        
        Args:
            submission_data (TYPE): Description
            idx (TYPE): Description
        """
        submission_data = self._ensure_batchsize_dimension(submission_data)
        batch_size = self.batch_size
        effective_batch_size = submission_data.shape[0]
        
        
        # restore original shape of files
        if self.ds.reduce:
            submission_data = self._restore_reduced_file_order(submission_data,
                                                               effective_batch_size)
        else:
            submission_data = self._restore_file_order(submission_data)
            
        
        # submission data should be int
        
#        submission_data = submission_data.type(torch.IntTensor)
        submission_data = submission_data.astype('int')
        
        
        # split submission data into files and then write to files       
        
        orig_ixs = self._get_orig_ix(idx, batch_size)
        file_ixs = list(orig_ixs//5)
        file_ixs_unique = list(set(file_ixs))
        
        last_file_ix = len(self.ds.target_file_paths) -1
#        print(self.ds.cities)
        
        
        for file_ix in file_ixs_unique:
            if file_ix > last_file_ix:
                print("Didn't write file number {}, there are not enough submission \
                      files for your data. Try setting the 'filter_test_times' \
                      flag in the dataloader".format(file_ix))
                continue
            arraystamps, filestamps = self._get_timestamps(file_ixs, file_ix, orig_ixs)
            
            test_filepath = self.ds.target_file_paths[file_ix]
            print("File: {}. I want to write arraystamps {} into filestamps {}. \
                  loader_ix: {}, batch_size: {}, effective_batch_size: {}".format(test_filepath, arraystamps, filestamps, idx, batch_size, effective_batch_size))                     
            submission_filename = test_filepath.replace(self.ds.target_root, 
                                                        self.submission_root)
            
#            print(self._get_timestamps(file_ixs, file_ix, orig_ixs))
            self._write_part_to_file(submission_data[arraystamps[0]:arraystamps[1], ...],
                                     submission_filename,
                                     filestamps)
            
    def fake_submission_writing(self, submission_data, idx):
        submission_data = self._ensure_batchsize_dimension(submission_data)
        batch_size = submission_data.shape[0]
        
        
        # restore original shape of files
        if self.ds.reduce:
            submission_data = self._restore_reduced_file_order(submission_data,
                                                               batch_size)
        else:
            submission_data = self._restore_file_order(submission_data)
            
        
        # submission data should be int
        
#        submission_data = submission_data.type(torch.IntTensor)
        submission_data = submission_data.astype('float')
        
        # split submission data into files and then write to files       
        
        orig_ixs = self._get_orig_ix(idx, batch_size)
        file_ixs = list(orig_ixs//5)
        file_ixs_unique = list(set(file_ixs))
        
        
        for file_ix in file_ixs_unique:
            
            filestamps, tstamps = self._get_timestamps(file_ixs, file_ix)
            test_filepath = self.ds.target_file_paths[file_ix]
                                 
            submission_filename = test_filepath.replace(self.ds.target_root, 
                                                        self.submission_root)
            
            
            
            
            debug_dict = {'submission_data': submission_data,
                          'filestamps0':filestamps[0],
                          'filestamps1':filestamps[1],
                          'submission_filename': submission_filename,
                          'tstamps': tstamps,
                    }
            return debug_dict
            

            
if __name__ == '__main__':  

    source_root = r"D:\Codes\traffic4cast\data\data_sample"
    target_root = "data"
    split_type='test'
    reduce = False
    batch_size = 3
    submission_root = os.path.join('submission', 'test')
    submission_root_red = os.path.join('submission', 'test_red')
    
    
    kwds_dataset = {'source_root': source_root,
                    'target_root': target_root, 
                    'split_type': split_type,
                    'reduce': reduce}
    kwds_loader = {'shuffle': False,
                   'num_workers':0,
                   'batch_size': batch_size}
   
    dataset_test = trafic4cast_dataset(**kwds_dataset)
    test_loader = torch.utils.data.DataLoader(dataset_test, **kwds_loader)
    
    kwds_dataset['reduce'] = True
    dataset_test_red = trafic4cast_dataset(**kwds_dataset)
    test_loader_red = torch.utils.data.DataLoader(dataset_test_red, **kwds_loader)
    
    sw = submission_writer(dataset_test,
                           submission_root=submission_root)
    
    sw_red = submission_writer(dataset_test_red,
                           submission_root=submission_root_red)
    
    
    
    # ---------------test again
    # get data
    for ix, (loaded_data, _) in enumerate(test_loader):
        break
    for ix_red, (loaded_data_red, _) in enumerate(test_loader_red):
        break
    assert ix == ix_red
    
    # get fake target data
    loaded_target = loaded_data [:,0:3,:,:,:]
    loaded_target_red = loaded_data_red[:,0:9,:,:]
    
    # get file ixs and load original data
    orig_ixs = sw._get_orig_ix(ix, batch_size)
    orig_ixs_red = sw_red._get_orig_ix(ix, batch_size)
    
    assert all(orig_ixs == orig_ixs_red)
        
    file_ixs = list(orig_ixs//5)
    file_ixs_unique = list(set(file_ixs))
    
    orig_data_list = []
    orig_target_list = []
    
    for test_data_ix in orig_ixs:
        
        (orig_filepath, tstamp_ix) = sw.ds.valid_test_clips[test_data_ix]
        orig_filepath = orig_filepath.replace(target_root, source_root)
        orig_data_full = read_hdf5_file(orig_filepath)
        
        original_data = orig_data_full[tstamp_ix:tstamp_ix+12,:,:,:]
        original_target = original_data[0:3,...]
        
        orig_data_list.append(original_data)
        orig_target_list.append(original_target)
    

    orig_data = np.stack(orig_data_list)
    orig_target = np.stack(orig_target_list)
    
    
     # test: _ensure_batchsize_dimension does not alter data
    loaded_data2 = sw._ensure_batchsize_dimension(loaded_data)
    loaded_data2_red = sw_red._ensure_batchsize_dimension(loaded_data_red)
    loaded_target2 = sw._ensure_batchsize_dimension(loaded_target)
    loaded_target2_red = sw_red._ensure_batchsize_dimension(loaded_target_red)
    
    assert np.sum(np.abs(np.squeeze(loaded_data2.numpy()) -
                         np.squeeze(loaded_data.numpy()))) == 0
                         
    assert np.sum(np.abs(np.squeeze(loaded_data2_red.numpy()) -
                         np.squeeze(loaded_data_red.numpy()))) == 0
                         
    assert np.sum(np.abs(np.squeeze(loaded_target2.numpy()) -
                         np.squeeze(loaded_target.numpy()))) == 0
                         
    assert np.sum(np.abs(np.squeeze(loaded_target2_red.numpy()) -
                         np.squeeze(loaded_target_red.numpy()))) == 0     
                         
                         
    loaded_data3 = np.squeeze(sw._restore_file_order(loaded_data2))
    
    assert np.sum(np.abs(np.squeeze(loaded_data3) -
                         np.squeeze(orig_data))) == 0
    
    loaded_target3 = np.squeeze(sw._restore_file_order(loaded_target2))
    loaded_target3_red = np.squeeze(sw_red._restore_reduced_file_order(
            loaded_target2_red, batch_size=batch_size))
    
    assert np.sum(np.abs(np.squeeze(loaded_target3) -
                         np.squeeze(orig_target))) == 0   
   
                         
    assert np.sum(np.abs(np.squeeze(loaded_target3_red) -
                         np.squeeze(orig_target))) == 0
                         
                         
                         
    print('Reshape pipeline test passed')                     
                        
        
        
    # Test partial file writing
    
    # write some test files
    for file_ix in file_ixs_unique:
        
        filestamps, tstamps = sw._get_timestamps(file_ixs, file_ix)        
        filestamps_red, tstamps_red = sw_red._get_timestamps(file_ixs, file_ix)
        
        test_filepath = sw.ds.target_file_paths[file_ix]
        test_filepath_red = sw_red.ds.target_file_paths[file_ix]
        assert test_filepath == test_filepath_red                    
        
        submission_filename = test_filepath.replace(sw.ds.target_root, 
                                                    sw.submission_root)
        submission_filename_red = test_filepath_red.replace(sw_red.ds.target_root, 
                                                    sw_red.submission_root)
        
        
        sw._write_part_to_file(loaded_target3[filestamps[0]:filestamps[1], ...],
                                 submission_filename,
                                 tstamps)
        sw_red._write_part_to_file(loaded_target3_red[filestamps_red[0]:filestamps_red[1], ...],
                                 submission_filename_red,
                                 tstamps_red)
        
    def fill_with_zeros(data):
        target_shape = (5, 3, 495, 436, 3)
            
        if data.shape == target_shape:
            return data
        
        delta = target_shape[0] - data.shape[0]
        
        return np.pad(data,((0,delta),(0,0),(0,0),(0,0),(0,0)))
        
            
    # reload test files and check if they where written correctly    
    for file_ix in file_ixs_unique:
    
        filestamps, tstamps = sw._get_timestamps(file_ixs, file_ix)        
        filestamps_red, tstamps_red = sw_red._get_timestamps(file_ixs, file_ix)
        
        test_filepath = sw.ds.target_file_paths[file_ix]
        test_filepath_red = sw_red.ds.target_file_paths[file_ix]
        submission_filename = test_filepath.replace(sw.ds.target_root, 
                                                    sw.submission_root)
        submission_filename_red = test_filepath.replace(sw_red.ds.target_root, 
                                                    sw_red.submission_root)
        
        reloaded_target = read_hdf5_file(submission_filename)
        reloaded_target_red = read_hdf5_file(submission_filename_red)
        
        target_this = loaded_target3[filestamps[0]:filestamps[1], ...]
        target_this_red = loaded_target3_red[filestamps_red[0]:filestamps_red[1],...]
        
        target_this2 = fill_with_zeros(target_this)
        target_this_red2 = fill_with_zeros(target_this_red)
        
        assert np.sum(np.abs(reloaded_target - target_this2)) == 0
        assert np.sum(np.abs(reloaded_target_red - target_this_red2)) == 0
        
    print('Input/Output test passed')

   
