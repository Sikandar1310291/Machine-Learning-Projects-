# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_model  = pickle.load(open(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\trained_model.sav" , "rb"))

input_data = (1,85,66,29,0,26.6,0.351,31)
#  input data convert to numpy array
input_data_array = np.asarray(input_data)
#  reshaping the input data
reshaped_data  =  input_data_array.reshape(1,-1)

#  predict our data
prediction = loaded_model.predict(reshaped_data)
print(prediction)

if (prediction[0] == 0):
    print("person is Non_Diabetic")
else:
    print("Person is Diabetic")