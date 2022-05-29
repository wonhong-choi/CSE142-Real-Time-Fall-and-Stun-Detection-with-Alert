# In[0]: import 및 경로 설정

# file name : index.py
# pwd : /project_name/app/main/index.py
import sys
from flask import Blueprint, request, render_template, flash, redirect, url_for, Response, session
from flask import current_app as current_app
from flask import jsonify
from flask_paginate import Pagination, get_page_args
import mediapipe as mp
from collections import deque
import numpy as np
import itertools
import threading
from datetime import datetime, date, timedelta
sys.path.append("C:/Users/pc/Desktop/flask/")
from normalize import *
import time
import cv2
import os
import csv

# In[1]: 모델 짬통
# 모델 파라미터 적는곳!

def get_path():
    path = "C:/Users/pc/Desktop/flask/app/static/img/occursVideo/"+datetime.today().strftime("%Y%m%d")
    return path

#  cctv = Cctv().getc(id);
#dir_list = [file for file in os.listdir(get_path()) if file.endswith('.mp4')]
#dir_list.sort(key=lambda x: int(x[:-4]))
    
#cctvList =[]
#for i,name in enumerate(dir_list):
#    temp ={ 'id':i+1, 'title':name }
#    cctvList.append(temp)    


# In[2]: blueprint 설정
main= Blueprint('main', __name__, url_prefix='/')
key = threading.Event()
key.set()

# In[]: 공유 데이터를 관리하는 클래스
class DataManager:
    def __init__(self,features):
        self.X_input = [[ 0 for i in range(len(features)*2)] for j in range(30)]
        self.label = "ready"
        self.data_d = [[ 0 for i in range(len(features)*2)] for j in range(90)]
        self.frame_d = None
        self.detected = False
        
    def setX_input(self, value):
        self.X_input = value
    
    def setlabel(self, value):
        self.label = value
        
    def setdata_d(self, value):
        self.data_d = value
        
    def setframe_d(self, value):
        self.frame_d = value
        
    def getX_input(self):
        return self.X_input
    
    def getlabel(self):
        return self.label[:]
    
    def getdata_d(self):
        return self.data_d
    
    def getframe_d(self):
        if self.frame_d:
            return self.frame_d.copy()
        return None

#features=[i for i in range(33)]
features=[2, 5, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
manager = DataManager(features)
# In[3]: route 설정

#@main.route('/home', methods=['GET','POST'])
#def index():
#    if request.method =='GET':
#        return render_template("index.html")
#    elif request.method =='POST':
#        userId = request.form['id']
#        userPassword = request.form['password']
#        
#        if userId == "admin" and userPassword == "dke244":
#            session['user'] = userId
#            return ''' <script> alert("안녕하세요 {}님"); location.href="/home" </script> '''.format(userId)
#        #redirect(url_for('main.home'))
#        else :
#            return ''' <script> alert("다시 시도해주세요!!"); location.href="/" </script> '''
        
@main.route('/register', methods=['GET'])
def register():
    return render_template("register.html")

@main.route('/', methods=['GET', 'POST'])
def index():
    #if 'user' in session:
        #if request.method == 'GET' :    
            #return render_template("home.html")
        
        #val = request.json.get(".result")
        #print(val)
        #return jsonify({"data": {"val": val}})
    #return render_template(url_for('main.index'))
    return render_template("home.html")

@main.route('/occurs/<id>/', methods=['GET'])
def occurs(id): 
        
    #  cctv = Cctv().getc(id);
    dir_list = [file for file in os.listdir(get_path()) if file.endswith('.mp4')]
    dir_list.sort(key=lambda x: int(x[:-4]))
        
    cctvList =[]
    for i,name in enumerate(dir_list):
        temp ={ 'id':i+1, 'title':name }
        cctvList.append(temp)  
        cctv = []
    
    for e in cctvList:
        if(e['id']==int(id)):
            cctv.append(e)
    
    today= {'day' : datetime.today().strftime("%Y%m%d")}
    
    if(request.method =='GET'):
        num_btn = request.args.get("temp")
        print(num_btn)
        print(type(num_btn))
        
        # 버튼을 사용자가 누르면 저장되었습니다 팝업창 출력하고 확인누르면 다시 occurs로 돌아가게 만들면 좋을것같음
        # 가능하다면 버튼도 확인, 수정 둘로 만들고, 수정을 누르면 자세를 입력할 수 있도록
        if num_btn != -1 and num_btn is not None:
            #90프레임, 라벨 각각 csv로 저장
            #추가학습데이터 폴더 만들어야 함
            
            today= date.today().strftime("%Y%m%d")
            f = open('./new_datas/'+ today +'/y_train'+id+'.csv', 'w', newline = '') # 생성하는 영상 이름별로
            wr = csv.writer(f)
            for i in range(3):
                wr.writerow(str(int(num_btn)-1))
            f.close
            
        
        return render_template("occurs.html", cctv=cctv, today=today)


def get_cctvs(offset=0, per_page=10, cctvList=[]):
    return cctvList[offset: offset + per_page]

@main.route('/occursList', methods=['GET', 'POST'])
def occursList():        
        
    #  cctv = Cctv().getc(id);
    dir_list = [file for file in os.listdir(get_path()) if file.endswith('.mp4')]
    dir_list.sort(key=lambda x: int(x[:-4]))
        
    cctvList =[]
    for i,name in enumerate(dir_list):
        temp ={ 'id':i+1, 'title':name }
        cctvList.append(temp)
        
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(cctvList)
    pagination_cctvs = get_cctvs(offset=offset, per_page=per_page, cctvList=cctvList)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap3')
    
    return render_template( 
        "occursList.html",
        cctvs=pagination_cctvs,
        page=page,
        per_page=per_page,
        pagination=pagination,)
    
#    elif(request.method =='POST'):
#        value = request.form('input')
#        return render_template("occurs.html",temp=value)

@main.route('/about', methods=['GET'])
def about():
    return render_template("about.html")

@main.route('/contact', methods=['GET'])
def contact():
    return render_template("contact.html")

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#@main.route('/logout')
#def logout():
#    session.clear()
#    return redirect(url_for('index'))


def gen_frames():
    global key
    global features
    global manager
    
    cap = cv2.VideoCapture(0)
    
    
    
    data_d = deque()
    frame_d=deque()  # 90프레임의 이미지 행렬을 저장하는 덱 >> 영상 저장에 쓰여야하므로 인공지능에 전달필요
    
    
    label = 'Ready'
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    
    X_input = deque()
    
    f = len(features)*2 # feature count
    buffer_size=90
    for i in range(30):
        X_input.append([0 for _ in range(f)])
    
    manager.setX_input(X_input)
    
    for buffer in range(buffer_size):
        data_d.append([0 for _ in range(f)])
        
        
    new_data = [0 for col in range(f)]
    

    BODY_PARTS = {0: "Nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer", 4: "right_eye_inner",
              5: "right_eye", 6: "right_eye_outer", 7: "left_ear", 8: "right_ear", 9: "mouth_left",
              10: "mouth_right", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
              15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 19: "left_index",
              20: "right_index", 21: "left_thumb", 22: "right_thumb", 23: "left_hip", 24: "right_hip", 
              25: "left_knee", 26: "right_knee", 27:"left_ankle", 28:"right_ankle", 29:"left_heel",
              30: "right_heel", 31: "left_foot_index", 32:"right_foot_index"}

    # For webcam input:
   
    cur_time=time.time()
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if key.is_set() == 0:
                cap.release()
                break
            
            success, image = cap.read()
            
            end_time=time.time()
            fps=int(1/(end_time-cur_time))
            cur_time=end_time
        
            if not success:
                print("Ignoring empty camera frame.")
                continue # If loading a video, use 'break' instead of 'continue'.
            
            
            frame_d.append(image)
            if len(frame_d)>buffer_size:
                frame_d.popleft()
                
            if manager.detected == False:
                manager.setframe_d(frame_d)
            
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_height, image_width, _ = image.shape
            results = pose.process(image)
        
        
            if not results.pose_landmarks:
                continue


            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            label = manager.getlabel()
            cv2.putText(image, label, (100,100), 0, 2, (0,0,255), 2, cv2.LINE_8, False)
            cv2.putText(image, str(fps), (400,100), 0, 2, (0,0,255), 2, cv2.LINE_8, False)
            #cv2.imshow('Pose Detection', image)
            
            idx=0
            for feature in features:
                new_data[2*idx] = results.pose_landmarks.landmark[feature].x
                new_data[2*idx+1] = results.pose_landmarks.landmark[feature].y
                idx+=1
            
            new_data=np.array(new_data)
            
            data_d.popleft()
            data_d.append(new_data.copy())
            manager.setdata_d(data_d)
            
            
            x_max=new_data[::2].max()
            x_min=new_data[::2].min()
            y_max=new_data[1::2].max()
            y_min=new_data[1::2].min()
            
            for i in range(len(new_data)):
                if i%2==0:
                    new_data[i]=(new_data[i]-x_min)/(x_max-x_min)
                else:
                    new_data[i]=(new_data[i]-y_min)/(y_max-y_min)
            
            
         
            #x_data = np.array(list(itertools.islice(data_d, buffer_size-30, buffer_size)))
            
            
            #X_input = normalize(x_data)
            X_input.append(new_data.copy())
            X_input.popleft()
            
            manager.setX_input(X_input)
            
            
            
            ret, buffer = cv2.imencode('.jpg', image)
            
            web_frame = buffer.tobytes()
            
            
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + web_frame + b'\r\n')  # concat frame one by one and show result
            
    cap.release()
            

