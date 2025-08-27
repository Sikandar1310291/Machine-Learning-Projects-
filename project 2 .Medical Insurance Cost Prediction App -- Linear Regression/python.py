# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:29:51 2025

@author: ma516
"""

import pickle
import numpy as np 
file_path = r"C:\Users\ma516\OneDrive\Desktop\medical insurance app\Model.sav"

loaded_model = pickle.load(open(file_path , 'rb'))
input_data = (21,1,25.800 ,0, 1 , 1)
array_data = np.asarray(input_data)
reshaped_data = array_data.reshape(1,-1)
prediction =  loaded_model.predict(reshaped_data)
print( prediction)
