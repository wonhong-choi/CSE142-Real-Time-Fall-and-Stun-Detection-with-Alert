# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:59:50 2022

@author: 권창덕
"""

from LSTM_tf2_User_version2 import *
import threading

# In[ ]:
if __name__ == '__main__':
    
    print("Prediction Start")
    
    model_name='m_91.9047619047619_2_time_step30_feature_cnt32_epochs90_batch_size96_input_size30_ver1.h5'
    
    model = load_model(model_name)
    
    d = deque()
    
    th1 = threading.Thread(target = Run, args = (model,))
    th1.start()
    
    app.run(host="0.0.0.0", port=5000)