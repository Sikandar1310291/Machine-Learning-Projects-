# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:31:21 2025

@author: ma516
"""

import pickle
import numpy as np
import pandas as pd
file = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Heart disease app\disease_app.sav"
loaded_model = pickle.load(open(file , 'rb'))


# input the variable
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
# Convert into array 
array_data = np.array(input_data)
# reshaped our data 
reshaped_data = array_data.reshape(1,-1)
#  predicted data
prediction = loaded_model.predict(reshaped_data)
print(prediction)
if(prediction == 0):
    print("Person Does not have a Heart Disease")
else:
    print("Person have Heart Disese")
