#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:18:29 2019

@author: neil
"""
import cv2
import numpy as np
import pandas as pd
import os
import time


def findPriorIndex(time, pred, real):
    index_pred = np.searchsorted(pred['time_sec'].values.tolist(), time) -1
    index_real = np.searchsorted(real['time_sec'].values.tolist(), time) -1
#    print("index_pred",index_pred)
#    print("index_real",index_real)
    return int(index_pred), int(index_real)

def putText(img,text,img_shape, color, org,save_dir,img_name):
    cv2.putText(img = img,
                text = text,
                org = org,
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.0015 * img_shape[0],
                color = color)
    cv2.imwrite(save_dir + "{}".format(img_name), img)

def loadData(subject_path):
    if not (subject_path[-1] =="/"):
        subject_path = subject_path + "/"
    label_path = [f for f in os.listdir(subject_path) if f.endswith('head_pose.csv')][0]
    frame_name_path = [f for f in os.listdir(subject_path) if f.endswith('frame_times.csv')][0]
    image_path = subject_path + [f for f in os.listdir(subject_path) if f.endswith('cropped_frames')][0] + '/'
    data = pd.read_csv(subject_path + label_path, delimiter=',')
#    print("\nlen_data:{}\n".format(np.shape(data)))
    pic_name = pd.read_csv(subject_path + frame_name_path, delimiter=',')
#    print(pic_name.head())
#    print("\nlen_pic_name:{}\n".format(np.shape(pic_name)))
    return data, pic_name, image_path
 
def dataProcessing(data, pic_name):
    # modify data:
    # print("data:\n",data.head())
    # print('pic_name:\n',pic_name.head())
    pic_name['time_sec'] = round(pic_name["time_sec"], 2)  # match label_time_sec
    pic_name.file = list(map(lambda x: x[-16:], pic_name.file))  # modify img_name
    # classify the label:drop time == 0 when it is not the threshold, and drop time_sec column for label
    if np.any(data.iloc[0, :-1] != 0):
        data = data.iloc[1:, :].reset_index(drop=True)
    #drop rows of label data that time is ahead of pic_ time
    if pic_name.loc[0,"time_sec"]>0:
        data = data[data["time_sec"]>pic_name.loc[0, "time_sec"]].reset_index(drop=True)
#    print('real_data.head()\n',data.head(5))
#    print('pic_name\n',pic_name.head())
    return data, pic_name 

def loadImg(data, index,img_path):
    name = data.loc[index, "file"]
    path = img_path + name 
#    print("path:",path)
    img = cv2.imread(path)
    return img

def convertToVideo(pathIn,pathOut):
    fps = 30
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn,f))]
    
    #sorting the file names properly
    files.sort(key = lambda x:int(x[-10:-4]))
    
    for i in range(len(files)):
        filename = pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
#    print("len_frames:",len(frame_array))
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    
    for i in range(len(frame_array)):
        #writing to a image array
        out.write(frame_array[i])
    cv2.destroyAllWindows()
    out.release()


def Real_Label(real_file):
    data = pd.read_csv(real_file)
    y = []
    for i in range(1, len(data)):
        if np.any(np.absolute(data.iloc[i, :3]) >= 0.025):  # or np.any(np.absolute(data.iloc[i, 3:-1]) >= 1):  # 1mm == 1
            y.append("Moving")
        else:
            y.append("Not_moving")
    time = np.array(data.loc[1:,'time_sec'])
    newDF = pd.DataFrame({"label":y,"time_sec":time})
#    print("newDF:\n",newDF.head())
    return newDF

#def pred_Label(real_file):
#    data = pd.read_csv(real_file)
#    y = []
#    for i in range(1, len(data)):
#        if np.any(np.absolute(data.iloc[i, :3]) >= 0.025) or np.any(np.absolute(data.iloc[i, 3:-1]) >= 1):  # 1mm == 1
#            y.append("Moving")
#        else:
#            y.append("Not_moving")
#    time = np.array(data.loc[1:,'time_sec'])
#    newDF = pd.DataFrame({"label":y,"time_sec":time})
##    print("newDF:\n",newDF.head())
#    return newDF

def overlay(subject_path, pred_file, real_file, save_dir_file, video_name = None):
    """
    
    This function is to put prediction into the frame and combined all these frame to a video
    For the total number of image is much larger than that of frames with labels
    If the time of that image matches predict_file time, put text in red
    else use the previous real value and prediction value in black
    
    input: 
        subject_path:the path to find the all data for that subjects
        pred_file: location of prediction.csv, each instance is [label,time_sec]  eg: ("not_moving", 1.3s)
        real_file: location of real.csv, each instance is [rotX,rotY,rotZ, transX,transY,transZ,time_sec]
    output:
        None
    
    """
    save_img = save_dir_file + 'Image/'
    if not os.path.exists(save_img):
        m = os.umask(0)
        os.makedirs(save_img)
        os.umask(m)
    #load frame by time from cropped.csv
    data, pic_name, image_path = loadData(subject_path = subject_path)
    #data preprocessing
    data, pic_name  = dataProcessing(data, pic_name)
    
    #read pred_file
    # pred = pd.read_csv(pred_file,index_col = 0)
    pred = pd.read_csv(pred_file)
    real_label = Real_Label(real_file)

    if 'Pred_label' not in pred.columns.values.tolist():
        temp = pred.copy()
        temp = temp.values
        temp_list = []
        for i in range(len(temp)):
            if np.any(np.absolute(temp[i, :3]) >= 0.025):  # 1mm == 1
                temp_list.append("Moving")
            else:
                temp_list.append("Not_moving")
        temp_value = np.array(temp_list)
        pred["Pred_label"] = temp_value
    print(pred.head(5))
    print("\n\nreal:\n",real_label.head(5))

    for i in range(len(pic_name)):
        if (i%100 == 0):
            print("frames process (already done overlay): {}/{}".format(i,len(pic_name)))
        img_name = pic_name.loc[i,'file']
        if pic_name.loc[i,'time_sec'] > data.iloc[-1,-1]:
#            print("stage 1:",i)
#            print("stage 1 time:",pic_name.loc[i,'time_sec'])
            break
        else:
            if pic_name.loc[i,'time_sec'] < pred.loc[0,'time_sec']:
#                print("stage 2:",i)
#                print("stage 2 time:",pic_name.loc[i,'time_sec'])
                continue
            else:
                if pic_name.loc[i,'time_sec'] in np.array(pred['time_sec']):
                    # print("stage 3:",i)
                    # print("stage 3 time:",pic_name.loc[i,'time_sec'])
                    #find its prior index in pred and data
                    if pic_name.loc[i,'time_sec'] ==pred.loc[0,'time_sec']:
                        index_pred, index_real = 0,int(len(data)-len(pred))
                    else:
                        index_pred, index_real = findPriorIndex(time = pic_name.loc[i,'time_sec'], real = data, pred = pred)     
                    img = loadImg(data = pic_name, index = i,img_path = image_path)
                    img_shape = img.shape
                    # real_value = "".join(str(round(i,2)) + " " for i in data.iloc[index_real,:])
                    real_value = [str(round(i,2)) for i in data.iloc[index_real,:]]
#                    print("type of pred:{}\n".format(type(pred.iloc[index_pred,0])),pred.head(5))
#                    print("type of real:{}\n".format(type(real_label.iloc[index_real,0])),real_label.head(5))
                    pred_value = "prediction:" + pred.loc[index_pred,"Pred_label"] + "     label:" + real_label.iloc[index_real,0]
                    # putText(img = img,text = real_value,img_shape = img_shape,
                                        #         color = (0,255,255), org = (0,260),save_dir = save_dir_file,
                                        #         img_name = img_name)
                    putText(img = img,
                            text = "RotX={},RotY={},RotZ={},Time={}".format(real_value[0],real_value[1],real_value[2],real_value[6]),
                            img_shape = img_shape,color = (0,255,255), org = (0,260),save_dir = save_img,
                            img_name = img_name)

                    putText(img = img,
                            text = "TransX={},TransY={},TransZ={},Time={}".format(real_value[3],real_value[4],real_value[5],real_value[6]),
                            img_shape = img_shape,color = (0,255,255), org = (0,280),save_dir = save_img,
                            img_name = img_name)

                    putText(img = img,text = pred_value,img_shape = img_shape, 
                            color = (0,255,255), org = (0,300),save_dir = save_img,
                            img_name = img_name)
                else:
                    #find its prior index in pred and data
                    # print("stage 4:",i)
                    # print("stage 4 time:",pic_name.loc[i,'time_sec'])
                    if pic_name.loc[i,'time_sec'] == pred.loc[0,'time_sec']:
                        index_pred, index_real = 0,int(len(data)-len(pred))
                    else:
                        index_pred, index_real = findPriorIndex(time = pic_name.loc[i,'time_sec'], real = data, pred = pred)
                    img = loadImg(data = pic_name, index = i,img_path = image_path)
                    img_shape = img.shape
                    # real_value = "".join(str(round(i,2)) + " " for i in data.iloc[index_real,:])
                    real_value = [str(round(i, 2)) for i in data.iloc[index_real, :]]
#                    print("real_value:",real_value)
                    pred_value = "Prediction:" + pred.loc[index_pred,"Pred_label"] + "     Label:" + real_label.iloc[index_real,0]

                    # putText(img = img,text = real_value,img_shape = img_shape,
                    #         color = (0,255,0), org = (0,260),save_dir = save_dir_file,
                    #         img_name = img_name)
                    putText(img=img,
                            text="RotX={},RotY={},RotZ={},Time={}".format(real_value[0], real_value[1], real_value[2],real_value[6]),
                            img_shape=img_shape, color=(0, 255, 0), org=(0, 260), save_dir=save_img,
                            img_name=img_name)
                    putText(img=img,
                            text="TransX={},TransY={},TransZ={},Time={}".format(real_value[3], real_value[4],real_value[5],real_value[6]),
                            img_shape=img_shape, color=(0, 255, 0), org=(0, 280), save_dir=save_img,
                            img_name=img_name)
                    putText(img = img,text = pred_value,img_shape = img_shape, 
                            color = (0,255,0), org = (0,300),save_dir = save_img,
                            img_name = img_name)
    
    #combined as a video
    convertToVideo(pathIn = save_img,pathOut = save_dir_file + '{}.mp4'.format(video_name))

def overlay_noLable(subject_path, pred_file, save_dir_file, video_name = None):
    save_img = save_dir_file + 'Image/'
    if not os.path.exists(save_img):
        m = os.umask(0)
        os.makedirs(save_img)
        os.umask(m)

    image_path = subject_path + [f for f in os.listdir(subject_path) if f.endswith('cropped_frames')][0] + '/'
    frame_times = subject_path + [f for f in os.listdir(subject_path) if f.endswith('frame_times.csv')][0]
    df_pred = pd.read_csv(pred_file,index_col=False)

    if 'Pred_label' not in df_pred.columns.values.tolist():
        temp = df_pred.copy()
        temp = temp.values
        temp_list = []
        for i in range(len(temp)):
            if np.any(np.absolute(temp[i, :3]) >= 0.025):  # 1mm == 1
                temp_list.append("Moving")
            else:
                temp_list.append("Not_moving")
        temp_value = np.array(temp_list)
        df_pred["Pred_label"] = temp_value

    print(df_pred.head(5))
    for i in range(len(df_pred)):
        pic_name = image_path + df_pred.loc[i, "Name"]
        img = cv2.imread(pic_name)
        img_shape = img.shape
        if df_pred.loc[i,'Pred_label'] == "Not_moving":
            putText(img=img, text=df_pred.loc[i,'Pred_label'], img_shape=img_shape,
                    color=(0, 255, 0), org=(0, 300), save_dir=save_img,
                    img_name=df_pred.loc[i,'Name'])
        else:
            putText(img=img, text=df_pred.loc[i,'Pred_label'], img_shape=img_shape,
                    color=(0, 255, 255), org=(0, 300), save_dir=save_img,
                    img_name=df_pred.loc[i,'Name'])

    convertToVideo(pathIn=save_img, pathOut=save_dir_file + '{}.mp4'.format(video_name))

    

def Display(subject_path = '/Users/neil/Desktop/summer_project/head_pose_sub_NC150_ses_20190410',
            pred_file = './prediction_subject.csv',real_file = './head_pose.csv',save_path = None, video_name = None, label =True):
    start = time.time()
    #call function
    if label==True:
        overlay(subject_path = subject_path,
                pred_file = pred_file, real_file =real_file,
                save_dir_file = save_path, video_name = video_name)
    else:
        overlay_noLable(subject_path=subject_path, pred_file=pred_file, save_dir_file=save_path, video_name=video_name)
    
    end = time.time()
    print("create a video time:{} seconds".format(end-start))


#path = '/Users/neil/Desktop/summer_project/head_pose_sub_NC150_ses_20190410/'
#head_pose = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
#Display(subject_path = path,pred_file = './prediction_subject.csv',real_file = path + head_pose)  
