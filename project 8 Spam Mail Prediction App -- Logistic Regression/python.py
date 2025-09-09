# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:02:24 2025

@author: ma516
"""
import pickle
import numpy as np



loaded_model = pickle.load(open(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\Mail.sav" , 'rb'))
tfidf = pickle.load(open(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\tfidf.sav"  , 'rb'))


# input Data
input_data = ["Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"]
# Transform of our input data
feature_selection = tfidf.transform(input_data)
# predict Our model
prediction = loaded_model.predict(feature_selection)
print(prediction)
if(prediction == 0):
    return("This Mail is Spam")
else:
    return("This Mail is Ham")