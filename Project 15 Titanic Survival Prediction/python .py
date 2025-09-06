# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 11:04:15 2025

@author: ma516
"""

import pickle
import numpy as np
filepath = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Project 15 Titanic Survival Prediction\Titanic.sav"
loaded_model = pickle.load(open(filepath , "rb"))

input_data = ( 1    ,   3  , 108  ,  1 , 22.000000  ,    1 ,     0  ,   523  , 7.2500 ,81     ,    2)
arrayed_data = np.array(input_data)
reshaped_data = arrayed_data.reshape(1,-1)
prediction = loaded_model.predict(reshaped_data)
print(prediction)
if(prediction == 0 ):
    print("Person is Died")
else:
    Print("Person is Survived")