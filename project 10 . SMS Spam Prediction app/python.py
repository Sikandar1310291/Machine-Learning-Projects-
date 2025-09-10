# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 23:46:05 2025

@author: ma516
"""

import pickle
import numpy as np
import pandas as pd
file_path1 = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Sms Spam prediction app\Sms.sav"
file_path2 = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Sms Spam prediction app\tfidf_SMS.sav"

loaded_model  =  pickle.load(open(file_path1 ,'rb' ))
tfidf = pickle.load(open(file_path2 , 'rb'))

input_data = ["Todays Vodafone numbers ending with 4882 are selected to a receive a å£350 award. If your number matches call 09064019014 to receive your å£350 award."]
Transformed_data = tfidf.transform(input_data)
prediction = loaded_model.predict(Transformed_data)
print(prediction)
if (prediction == 0 ):
    return("Sms is Ham")
else:
    return("Sms is Spam")