# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:54:14 2019

@author: henry
"""

import glob
import os
from NeurIPS2019traffic4cast.utils.h5shape import load_test_file

submission_root = r"E:\Programming\traffic4cast\submission\persistence_forecast"
 
file_paths = glob.glob(os.path.join(submission_root, '*', '*','*.h5'))   

for file_path in file_paths:
    file = load_test_file(file_path)
    
    if file.shape != (1, 5, 3, 495, 436, 3):
        print(file_path)
        


