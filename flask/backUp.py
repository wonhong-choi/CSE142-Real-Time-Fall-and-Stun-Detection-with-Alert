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
from app.index import *
from app.templates import *
from app.static import *
from datetime import datetime


LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4"
]
# In[2]:
def Run(model):
    global data
    global label
    data = [[[0 for col in range(46)] for row in range(30)]]
    label = "Ready"
    count = 0
    lbList = ["Standing", "Sitting", "Folding", "Stun", "Fall"]

    while True:
        acc_cnt = 0
        #start = time.time()
        predictions = model.predict(data)
        conf = int(predictions.argmax())
        #end = time.time()
        #print(end - start)

        #오리지널
        label = lbList[conf]

        '''
        #룰
        if conf == 0:
            count = 0
            label = lbList[conf]
        elif conf == 1:
            count = 0
            label = lbList[conf]
        elif conf == 2:
            count = 0
            if label == "Standing":
                label = lbList[conf]
        elif conf == 3:
            count+=1
            if label == "Sitting" and count == 5:
                count = 0
                label = lbList[conf]
        else:
            count+=1
            if count == 5:
                count = 0
                label = lbList[conf]
        '''


# In[3]:
def Live_Cam(features):
    global data
    global label
    global d
    global rayid
    
    rayid = 0
    d = deque()
    
    label = 'Ready'
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    
    frame_d=deque()
    
    f = len(features)*2 # feature count
    buffer_size=90
    
    #d=deque()
    for buffer in range(buffer_size):
        d.append([0 for _ in range(f)])
        
        
    new_data = [0 for col in range(f)]
    
    data = [[[0 for col in range(f)] for row in range(30)]]

    BODY_PARTS = {0: "Nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer", 4: "right_eye_inner",
              5: "right_eye", 6: "right_eye_outer", 7: "left_ear", 8: "right_ear", 9: "mouth_left",
              10: "mouth_right", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
              15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 19: "left_index",
              20: "right_index", 21: "left_thumb", 22: "right_thumb", 23: "left_hip", 24: "right_hip", 
              25: "left_knee", 26: "right_knee", 27:"left_ankle", 28:"right_ankle", 29:"left_heel",
              30: "right_heel", 31: "left_foot_index", 32:"right_foot_index"}

    # For webcam input:
    cap = cv2.VideoCapture(0)

    '''
    w= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    '''
    fourcc=cv2.VideoWriter_fourcc(*'DIVX')
    cap_out=cv2.VideoWriter('output.mp4',fourcc,15,(640,480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,2000)
   
    cur_time=time.time()
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if cap_out is None:
                cap_out = cap_out = cv2.VideoWriter('output.mp4',fourcc,15,(640,480))
            success, image = cap.read()
            
            #넘겨줄이미지
            if success:
                #셋 프레임
                q.put_nowait(image)
            fps=int(1/(end_time-cur_time))
            cur_time=end_time
        
            if not success:
                print("Ignoring empty camera frame.")
                continue # If loading a video, use 'break' instead of 'continue'.
            
            
            frame_d.append(image)
            if len(frame_d)>buffer_size:
                frame_d.popleft()
            #frame_d.popleft()
            
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_height, image_width, _ = image.shape
            results = pose.process(image)
        
        
            if not results.pose_landmarks:
                continue
            #print(results.pose_landmarks.landmark)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.putText(image, label, (100,100), 0, 2, (0,0,255), 2, cv2.LINE_8, False)
            cv2.putText(image, str(fps), (400,100), 0, 2, (0,0,255), 2, cv2.LINE_8, False)
            cv2.imshow('Pose Detection', image)
            
            idx=0
            for feature in features:
                new_data[2*idx] = results.pose_landmarks.landmark[feature].x * image_width
                new_data[2*idx+1] = results.pose_landmarks.landmark[feature].y * image_height
                idx+=1

            d.popleft()
            d.append(new_data)
            
            x_data = np.array(list(itertools.islice(d, buffer_size-30, buffer_size)))
            
            data = normalize(x_data)
            
            interrupt=cv2.waitKey(5)
            if interrupt == 27: # if pressed 'esc'
                os.kill
                break
            elif interrupt==32: # elif pressed 'space bar'
                while frame_d:
                    cap_out.write(frame_d.popleft())
                    #frame_d.append(None)
                cap_out.release()
                cap_out=None
                #cap_out=cv2.VideoWriter('output.mp4',fourcc,15,(640,480))
                #print(d)
            
    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()





# In[ ]:
def normalize(x_data):
    cur = pd.DataFrame(x_data)

    x_max=max(cur.iloc[:,::2].max())
    x_min=min(cur.iloc[:,::2].min())
    y_max=max(cur.iloc[:,1::2].max())
    y_min=min(cur.iloc[:,1::2].min())

    X_data = []
    for row in range(len(cur)):
        temp=[]
        for col in range(len(cur.iloc[row])):
            if col%2==0:
                temp.append((cur.iloc[row,col]-x_min)/(x_max-x_min))
            else:
                temp.append((cur.iloc[row,col]-y_min)/(y_max-y_min))

        X_data.append(temp)
    if len(cur) > 30:
        X_data = np.array((np.split(np.array(X_data), len(cur)/30)))
        return X_data
    X_data = [X_data]
    return np.array(X_data)
    
# In[5]:
def main(web_id):
    global d
    ver=0 # version 수정해야됨
    

    while True:
        '''
        #new_label=list(map(int,input('enter label:').split(' '))) # label 3개, 애매하다 == -1, [-2 -2 -2 ] == 종료
        
        # exit
        #if new_label[-1]==new_label[-2]==new_label[-3]==-2:
            print('bye')
            break
        #net_idx=[]
        #net_data=[]
        #for i in range(len(new_label)):
            if new_label[i]>=0:
                net_idx.append(i)
                net_data.append(d[i*30:i*30+30])
        #net_data=[]
        '''
        #print(test_num)
        # 겟 버튼
        
        test_num = v.value
        if test_num != 0:
            print(test_num)
            if test_num == -1:
                # 셋 논
                v.value = 0
                continue
            else:
                #90프레임, 라벨 각각 csv로 저장
                #추가학습데이터 폴더 만들어야 함
                df = pd.DataFrame(list(d))
                df.to_csv('./추가학습데이터/X_train.csv', mode = 'a+')
                f = open('./추가학습데이터/y_train.csv', 'a', newline = '')
                wr = csv.writer(f)
                for i in range(3):
                    wr.writerow(test_num)
                f.close  
                v.value = 0
                
                
        '''       
        # 시간 읽어와서 시간이 되면 멈추고 모델 학습 진행
        current_time = datetime.now().time()
        if current_time.hour == 20:
            key.clear()
            X_train = load_X('./추가학습데이터/X_Train.csv')
            y_train = load_y('./추가학습데이터/y_train.csv')
            y_train = to_categorical(y_train, num_classes=5)
            model.fit(X_train, y_train, epochs = 30, batch_size = 16, verbose = 1, shuffle = True)
            ver += 1
            model.save('hidden2_batch16_epochs30_ver' + str(ver) + '.h5')
            key.set()
        '''
