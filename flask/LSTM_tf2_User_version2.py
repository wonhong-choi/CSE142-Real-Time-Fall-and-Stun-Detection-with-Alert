#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import tensorflow as tf
from sklearn import metrics
import time
import os
import pandas as pd
import cv2
import mediapipe as mp
import keras
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from collections import deque
import itertools
import csv
from app import app
from app.templates import *
from app.static import *
from app.index import *
from normalize import normalize
from datetime import datetime, date, timedelta
import requests


LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

ver = 0

# In[2]:
def Run(model):
    global manager
    global features
    global key
    
    time.sleep(15)
    
    normal_count=0 # standing or sitting or folding detection count
    faint_count = 0 # fall or stun detection count
    
    lbList = ["Standing", "Sitting", "Folding", "Stun", "Fall"]
    
    

    recoding_count=0
    
    message_sent=False
    is_trained=False # False가 기본임 지금 테스트 상태
    
    ans = 0
    
    while True:
        time.sleep(0.3)
        acc_cnt = 0
        data = np.array([np.array(manager.getX_input())])
        #print(data)
        if data is not None:
            predictions = model.predict(data)
            #print(predictions)
            #print(sum(predictions))
            conf = int(predictions.argmax())
            
            if conf == 3 and ans == 1:
                ans = 3
            elif conf !=3:
                ans = conf
            
            if ans>= 3:
                faint_count+=1
                normal_count=0
            
            else:
                faint_count=0
                normal_count+=1
            
            label = lbList[ans]
                
            manager.setlabel(label)
        
        if faint_count >= 3 and message_sent==False:
            
            # send 알람
            
            try:
            
                TARGET_URL = 'https://notify-api.line.me/api/notify'
                TOKEN = 'ltk1XkCurCx3ybRmAZxO0UCYm4ZvV9QDkkl1f7bZX75'
            
                with open('./test.png', 'rb') as file:
                    response = requests.post(
                        TARGET_URL,
                        headers={
                            'Authorization': 'Bearer ' + TOKEN
                        },
                        data={
                            'message': 'Accident 발생\nhttp://172.19.92.143/5000/'
                        },
                        files = {
                            'file': file
                        }
                    )
            
                print(response.text)
            except Exception as ex:
                print(ex)
            
            
            manager.detected = True
            
            recoding_count+=1
            fourcc=cv2.VideoWriter_fourcc(*'H264')
            
            blackbox=cv2.VideoWriter(get_path()+'/'+str(recoding_count)+'.mp4',fourcc,30,(640,480))
            
            frame_d = manager.getframe_d()
            while frame_d:
                blackbox.write(frame_d.popleft())
   
            blackbox.release()
            
            today= date.today().strftime("%Y%m%d")
            video_path=get_path()[:-8]+today+'/'
            
            try:
                if not os.path.exists('./new_datas/' + today):
                    os.makedirs('./new_datas/' + today)
            except OSError:
                print("폴더 못만듬ㅋ")
                
            X_d = manager.getdata_d()
            X_d = normalize(X_d)
            X_df = pd.DataFrame(X_d)
            X_df.to_csv('./new_datas/' +today +'/X_train' + str(recoding_count) +'.csv', header = False, index = False)
            
            message_sent=True
            
            
        elif normal_count>=3:
            manager.detected = False
            message_sent=False
        
        
        current_time = datetime.now().time()
        if current_time.hour == 0 and is_trained==False:
            key.clear()
            recoding_count = 0
            #폴더 새로생성
            try:
                if not os.path.exists(get_path()):
                    os.makedirs(get_path())
            except OSError:
                print("폴더 못만듬ㅋ")
            
            add_train(model)
            
            key.set()
            
            is_trained=True
        elif current_time.hour==1:
            is_trained=False
            

# In[]:
def load_X(X_name):
    file = open(X_name, 'r')
    time_step=30
    X_ = np.array(
    [elem for elem in [
        row.split(',') for row in file
    ]],
    dtype=np.float32)
    file.close()
    blocks = int(len(X_) / time_step)
    X_ = np.array(np.split(X_, blocks))

    return X_

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.float32
    )
    
    
    file.close()
    return y_
# In[]:
def add_train(model):
    global key
    global ver
    global features
    
    yesterday=(date.today()-timedelta(days=1)).strftime("%Y%m%d")
    csv_path='./new_datas/' +yesterday + '/'
    detected_X_csv=[add_csv for add_csv in sorted(os.listdir(csv_path)) if add_csv.startswith('X_train')]
    detected_X_csv.sort()
    
    
    X_train=pd.DataFrame(columns=[i for i in range(len(features)*2)])
    y_train=pd.DataFrame(columns=[0])
    
    for i in range(1,len(detected_X_csv)+1):
        
        temp_y = pd.read_csv(csv_path + 'y_train'+str(i)+'.csv',header=None)
        if temp_y.iloc[2,0] == '-':
            continue
        temp_x = pd.read_csv(csv_path + 'X_train'+str(i)+'.csv',header=None)
        
        y_train=y_train.append(pd.read_csv(csv_path + 'y_train'+str(i)+'.csv',header=None),ignore_index=True)
        X_train=X_train.append(pd.read_csv(csv_path + 'X_train'+str(i)+'.csv',header=None),ignore_index=True)
    
             
    if len(X_train) < 30:
        print("검출된 영상이 없습니다")
        return
    
    X_train.to_csv(csv_path + 'X_train.csv', header = False, index = False)
    y_train.to_csv(csv_path + 'y_train.csv', header = False, index = False)  
    
            
    
    X_train = load_X(csv_path + 'X_train.csv') # X_train을 저장한 영상에서 뽑아오도록 수정필요
    y_train = load_y(csv_path + 'y_train.csv')
    y_train = to_categorical(y_train, num_classes=5)
    model.fit(X_train, y_train, epochs = 90, batch_size = 32, verbose = 0, shuffle = True) #이건 제공하는 모델 파라미터랑 동기화
    ver += 1
    model.save('add_train_model_ver' + str(ver) + '.h5') #이름 규칙 정해야함
     
    print("재학습완료")
         
     