# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:11:48 2022

@author: 권창덕
"""
import pandas as pd
import numpy as np
# In[ ]:
def normalize(x_data):
    cur = pd.DataFrame(x_data)

    X_data = []
    for row in range(len(cur)):
        temp=[]
        x_max = cur.iloc[row,::2].max()
        y_max = cur.iloc[row,1::2].max()
        x_min = cur.iloc[row,::2].min()
        y_min = cur.iloc[row,1::2].min()
        for col in range(len(cur.iloc[row])):
            if col%2==0:
                temp.append((cur.iloc[row,col]-x_min)/(x_max-x_min))
            else:
                temp.append((cur.iloc[row,col]-y_min)/(y_max-y_min))

        X_data.append(temp)
        
    return np.array(X_data)
