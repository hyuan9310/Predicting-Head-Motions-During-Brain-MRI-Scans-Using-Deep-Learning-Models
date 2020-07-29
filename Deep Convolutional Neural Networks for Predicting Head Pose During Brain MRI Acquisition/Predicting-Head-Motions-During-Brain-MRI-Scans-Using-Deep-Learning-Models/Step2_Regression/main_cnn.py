#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:11:36 2019

@author: neil
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import cv2
import random
import os
import sys


# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
import tensorflow as tf

import datetime

###########################################
class subject(object):
    def __init__(self,feature_path, label_path, img_path, group_size):
        self.feature_path = feature_path
        self.label_path = label_path
        self.img_path = img_path
        self.group_size = group_size

    def data_loading(self):
        """
        with_intermediate == True:means it includes the intermediate frames(38 frames between two frames with labels)
        load data
        get x and y for the general use
        :return: X and Y;
        X: X.shape:(nums_of_instance,img_row,img_col,channel)
        Y: Y.shape:(nums_of_instance,1)
        return: data
        """
        data = pd.read_csv(self.label_path, delimiter=',')
        pic_name = pd.read_csv(self.feature_path, delimiter=',')

        # modify data:
        if not np.any(pic_name.time_sec == 0):
            pic_name.time_sec = round(pic_name.time_sec, 2)  # match label_time_sec
        pic_name.file = list(map(lambda x: x[-16:], pic_name.file))  # modify img_name

        # classify the label:drop time == 0 when it is not the threshold, and drop time_sec column for label
        if np.any(data.iloc[0, :-1] != 0):
            data = data.iloc[1:, :].reset_index(drop=True)

        #drop rows of label data that time is ahead of pic_ time
        if pic_name.loc[0,"time_sec"]>0:
            data = data[data["time_sec"]>=pic_name.loc[0, "time_sec"]].reset_index(drop=True)

        label = np.array(data.iloc[:, :-1]) #without time
        time = np.array(data["time_sec"])
        #assume the first row is initial frame, the time at that frame may not be 0
        Y = label - label[0]

        y = [0]
        for i in range(1, len(Y)):
            if np.any(np.absolute(Y[i, :3]) >= 0.025):  #any pose >=0.025, means move
                y.append(1)
            else:
                y.append(0)
        Y = np.concatenate((np.array(y).reshape(len(y),1), time.reshape(len(time),1)), axis=1)

        # get feature
        new_df = pd.merge(pic_name, data, how='right', on='time_sec')
        pic_list = []
        for name in new_df.file:
            name = self.img_path + name
            img = cv2.imread(name, 0)
            img = cv2.resize(img, (224, 224))
            img = img_to_array(img)
            print("img_to_array:",np.shape(img))
            pic_list.append(img)
        X = np.array(pic_list)
        return X, Y

    def batch_feature_6frame(self, x, y, group_size):
        """
        input:  x: array(frames,img_row,img_col,size)
                y: array(frames,2)

        output: x_batch:(None,img_row, img_col,size)
                y_batch:(None,2)
                size(y_batch) = size(x_batch)

        set size =2 :input will be (fn-1-f0, fn-f0, fn - fn-1)to predict label ln;  total 3 frames as a batch,next group
                     is (fn-f0,fn+1 - f0,fn+1 - fn), a time series features
        """
        y_batch = y[group_size:]
        x_batch = []
        for i in range(1, len(x) - group_size + 1):
            feature = [x[0]]
            # feature = []
            for m in range(i, i + group_size):
                #append:fn-1,fn
                feature.append(x[m])
            for k in range(i, i + group_size):
                # append: fn-1-f0,fn-f0
                feature.append(x[k] - x[0])
            for j in range(i, i + group_size - 1):
                #append: - fn-1 + fn
                feature.append(x[j + 1] - x[j])
            feature = np.array(feature).reshape(x.shape[1], x.shape[2], -1)
            # create one batch as one input feature, shape = [256,256,3*(size-1)]
            x_batch.append(feature)
        return np.array(x_batch).reshape(len(x_batch), x.shape[1], x.shape[2],-1), np.array(y_batch).reshape(-1, 2)

    def batch_feature(self, x, y, group_size):
        """
        input:  x: array(frames,img_row,img_col,size)
                y: array(frames,2)

        output: x_batch:(None,img_row, img_col,size)
                y_batch:(None,2)
                size(y_batch) = size(x_batch)

        set size =2 :input will be (fn-1-f0, fn-f0, fn - fn-1)to predict label ln;  total 3 frames as a batch,next group
                     is (fn-f0,fn+1 - f0,fn+1 - fn), a time series features
        """
        y_batch = y[group_size:]
        x_batch = []
        for i in range(1, len(x) - group_size + 1):
            feature = []
            # feature = []
            for k in range(i, i + group_size):
                # append: fn-1-f0,fn-f0
                feature.append(x[k] - x[0])
            for j in range(i, i + group_size - 1):
                #append: - fn-1 + fn
                feature.append(x[j + 1] - x[j])
        #     feature = np.array(feature).reshape(x.shape[1], x.shape[2], -1)
        #     # create one batch as one input feature, shape = [256,256,3*(size-1)]
        #     x_batch.append(feature)
        # return np.array(x_batch).reshape(len(x_batch), x.shape[1], x.shape[2],-1), np.array(y_batch).reshape(-1, 1)
                feature = np.array(feature).reshape(x.shape[1], x.shape[2],-1)
            # create one batch as one input feature, shape = [256,256,3*(size-1)]
            x_batch.append(feature)
        return np.array(x_batch).reshape(len(x_batch),x.shape[1], x.shape[2],-1), np.array(y_batch).reshape(-1, 2)

    def train_data(self):
        # data preprocessing
        x, y = self.data_loading()
        x = x / 255
        # batch_feature:
        x, y = self.batch_feature(x, y, group_size=self.group_size)
        # time = y[:,-1]
        # x, y = self.batch_feature_6frame(x, y, group_size=self.group_size)
        # y = np.concatenate((y.reshape(len(y),-1),time.reshape(len(time),-1)),axis=1)
        print("\n\nthe length of x and y after batching:\n{}\n{}\n\n".format(np.shape(x),np.shape(y)))
        return x, y

class cnn_model(object):
    """
    input image size:(img_row,img_col,1)
    default times = 3
    default epochs =1000
    default batch_size = 156
    """

    def __init__(self, batch_size=156, epochs=1000, class_weight = None, input_shape=None):
        """

        :param batch_size: put the number of batch_size to train model each time
        :param epochs: int;
        :param class_weight: dictionary
        :param input_shape: tuple, shape of feature
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.class_weight = class_weight
        # define CNN model
        print('This is self-defined cnn model')
        cnn = Sequential()
        cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',
                       input_shape=self.input_shape))  # input is a greyscale pic
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=2))  # strides: If None, it will default to pool_size.
        cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(512, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(256, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(256, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(128, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(64, activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.5))
        cnn.add(Dense(1, activation=tf.nn.sigmoid))
        cnn.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
        print(cnn.summary())
        self.model = cnn
        pass

    def fitting(self, xtrain, ytrain, xVal=[], yVal=[]):
        """
        :param xtrain:
        :param ytrain:
        :param xVal:
        :param yVal:
        :return:
        """
        es = EarlyStopping(min_delta=0.001, patience=200, mode='min', monitor='val_loss')
        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)

        if len(xVal) == 0 and len(yVal) == 0:
            # use GPU
            gpu_no = 0
            with tf.device("/gpu:" + str(gpu_no)):
                print("this is to run gpu")
                sys.stdout.flush()
                self.model.fit(xtrain, ytrain,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_split=0.3,
                           class_weight=self.class_weight,
                           verbose=2,
                           callbacks=[es, rp])
        else:
            # use GPU
            gpu_no = 0
            with tf.device("/gpu:" + str(gpu_no)):
                print("this is to run gpu")
                sys.stdout.flush()
                self.model.fit(xtrain, ytrain,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_data=(xVal, yVal),
                               class_weight=self.class_weight,
                               verbose = 2,
                               callbacks=[es, rp])

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        # print('\ntest loss', loss)
        # print('\naccuracy', accuracy)

        y_pred = self.model.predict(x_test)
        y_pred = list(map(lambda x: 1 if x[0] >= 0.5 else 0, y_pred))
        y_pred = np.array(y_pred).reshape(-1, 2)
        count_1 = 0
        count_0 = 0
        for i in range(len(y_test)):
            if y_pred[i] == 1 and y_test[i] == 1:
                count_1 += 1
            if y_pred[i] == 0 and y_test[i] == 0:
                count_0 += 1
        # print('\ncorrect_count_0\n', count_0, '\ncorrect_count_1\n', count_1)
        # print('\nnumber of class 0 & class 1(test)\n', np.sum(y_test, axis=0))
        return (count_0,count_1,np.sum(y_test),len(y_test) - np.sum(y_test), accuracy, loss)

    def predict(self, xtest):
        y_pred = self.model.predict(xtest)
        # print("line237:{}\n\n".format(y_pred[:5]))
        y_pred = list(map(lambda x: [1, 0] if x[0] >= 0.5 else [0, 1], y_pred))
        # print("line 239:y_pred\n",y_pred)
        y_pred = np.array(y_pred)
        # print("line 241:y_pred\n", y_pred)
        return y_pred

    def get_model(self):
        return self.model

    def save_model(self,name):
        self.model.save(name)

def split_train_test(x,y,test_sample):
    """
    Note: since the features in the batch exist in its prior and post batches, so it may lead to testing features trained for the model.
    Thus, I delete the prior and post batch of each test data
    :param x:
    :param y:
    :param test_sample: randomly select the number of the test data
    :return: test and training datat
    """
    total_index = list(range(len(y)))
    test_index = random.sample(total_index, test_sample)

    array_test = np.array(test_index)
    del_index = np.unique(np.concatenate((array_test - 1, array_test, array_test + 1)))
    del_index = del_index[del_index >= 0]
    train_index = np.delete(np.arange(len(y)), del_index)

    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    return x_train, y_train, x_test, y_test

def data_load_subjects(subject_list,test_sample,group_size):
    #x_train y_train is all the data before split and del test
    x_train = []
    y_train = []
    x_train_dict = {}
    y_train_dict = {}
    x_test = {}
    y_test = {}
    for subject_id in subject_list:
        print("This is load subject {}.\n".format(subject_id))
        #path = "/u/erdos/students/hyuan11/MRI/gpfs/data/epilepsy/mri/hpardoe/head_pose_datasets/sub-NC{}_head_pose_data_zhao/".format(
#            subject_id)
        path = "/u/erdos/csga/jzhou40/MRI_Project/sub-NC{}_head_pose_data_zhao/".format(subject_id)
        # os.listdir: returns a list containing the names of the entries
        label_name = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
        feature_name = [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
        img_dir_name = [f for f in os.listdir(path) if f.endswith('cropped_frames')][0]
        # print("img_dir_name", img_dir_name)

        feature_path = path + feature_name
        label_path = path + label_name
        img_path = path + img_dir_name + "/"
        Subject = subject(feature_path=feature_path, label_path=label_path, img_path=img_path, group_size=group_size)
        x,y = Subject.train_data()
        xTrain, yTrain, xTest, yTest = split_train_test(x,y,test_sample)

        if len(x_train) == 0:
            x_train = xTrain
            y_train = yTrain
        else:
            x_train = np.concatenate((x_train, xTrain), axis=0)
            y_train = np.concatenate((y_train, yTrain), axis=0)
        #for prediction use
        x_train_dict[subject_id] = xTrain
        y_train_dict[subject_id] = yTrain

        x_test[subject_id] = xTest
        y_test[subject_id] = yTest

    print("DP_Line278: \nlen(x_train):{}\nlen(y_train):{}\n".format(len(x_train), len(y_train)))
    return x_train, y_train, x_test, y_test, x_train_dict, y_train_dict

def data_load_subjects_LOO(group_size = 2, test_subject = None,subject_list = None):
    assert (test_subject in subject_list)
    # print("test_subject_list",test_subject_list)
    # print("test_subject",test_subject)
    #for training data
    x_train =[]
    y_train = []
    x_train_dict = {}
    y_train_dict = {}

    for subject_id in subject_list:
        print("This is load subject {}.\n".format(subject_id))
        path = "/u/erdos/csga/jzhou40/MRI_Project/sub-NC{}_head_pose_data_zhao/".format(
            subject_id)
        #os.listdir: returns a list containing the names of the entries
        label_name = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
        feature_name= [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
        img_dir_name = [f for f in os.listdir(path) if f.endswith('cropped_frames')][0]
        print("img_dir_name", img_dir_name)

        feature_path = path + feature_name
        label_path = path + label_name
        img_path = path + img_dir_name + "/"

        Subject = subject(feature_path=feature_path, label_path=label_path, img_path=img_path, group_size=group_size)
        if subject_id == test_subject:
            print("This is to load testing subject {}.\n".format(test_subject))
            x_test, y_test = Subject.train_data()
            # print("len(x_test):{}\nlen(y_test):{}\n".format(len(x_test), len(y_test)))
        else:
            x, y = Subject.train_data()
            if len(x_train) == 0:
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate((x_train,x), axis=0)
                y_train = np.concatenate((y_train,y),axis = 0)
            x_train_dict[subject_id] = x
            y_train_dict[subject_id] = y

    print("func_data_load_subjects_LOO:\nxtrain.shape:{}\nytrain.shape:{}".format(x_train.shape,y_train.shape))
    print("dictionary_keys:{}".format(y_train_dict.keys()))
    print("xtest.shape:{}\ny_test.shape:{}\n".format(x_test.shape,y_test.shape))

    return x_train, y_train, x_test, y_test, x_train_dict, y_train_dict

def main_allSj(save_dir_file, group_size, subject_list,test_sample,batch,epochs,zero_weight):
    x_train, y_train, x_test, y_test, x_train_dict, y_train_dict = data_load_subjects(subject_list = subject_list, test_sample = test_sample, group_size = group_size)
    # save real value by folders
    real_value_path = save_dir_file + "Real_value_test/"
    if not os.path.exists(real_value_path):
        m = os.umask(0)
        os.makedirs(real_value_path)
        os.umask(m)

    # for keys, values in y_train_dict.items():
    #     y_train_df = pd.DataFrame(values, columns=['Not_moving', 'moving'])
    #     print("This is the path to save real value:\n" + real_value_path + "{}.csv".format("{}_train".format(keys)))
    #     y_train_df.append({"Not_moving": np.sum(y_train_df, axis=0)[0], "moving": np.sum(y_train_df, axis=0)[1]},
    #                   ignore_index=True)
    #     y_train_df.to_csv(real_value_path + "Real_{}.csv".format(keys))

    # build mode:
    zero_weight = zero_weight

    print("This is zero_weight of {}".format(zero_weight))
    model = cnn_model(batch_size=batch, epochs=epochs, class_weight={0: zero_weight, 1: 1},
                      input_shape=x_train.shape[1:])
    model.fitting(x_train, y_train[:,:-1])
    # save model:
    # model.save_model(name=save_dir_file + "{}_classification.h5".format(YMD))

    result = pd.DataFrame(
        columns=["test_subject", "correct_not_moving", "correct_moving", "total_not_moving", "total_moving"])
    result_train = pd.DataFrame(
        columns=["train_subject", "correct_not_moving", "correct_moving", "total_not_moving", "total_moving"])
    for num in subject_list:
        print("\nThis is testing subject {}\n".format(num))
        count_0, count_1, total_0, total_1, accuracy, loss = model.evaluate(x_test[num], y_test[num])
        print("(subject_id, accuracy,0_accu, 1_accu):{},{},{},{}".format(int(num), round(accuracy, 2),round(count_0/total_0,2), round(count_1/total_1,2)))
        # y_pred_test = model.predict(x_test[num])
        print('\ncorrect_count_0:{}'.format(count_0), '\ncorrect_count_1:{}'.format(count_1))
        print('The number of class not_moving & class moving of testing subject {0}: {1},{2}\n'.format(int(num), int(total_0),int(total_1)))
        result = result.append(
                {'test_subject': int(num), 'correct_not_moving': count_0, 'correct_moving': count_1,
                 'total_not_moving': total_0,
                 'total_moving': total_1},
                ignore_index=True)

        print("\nThis is training subject {}\n".format(num))
        count_0, count_1, total_0, total_1, accuracy, loss = model.evaluate(x_train_dict[num], y_train_dict[num])
        # print("subject_id, accuracy:{},{}".format(num, round(accuracy, 2)))
        # y_pred_test = model.predict(x_train_dict[num])
        print('\ncorrect_count_0:{}'.format(count_0), '\ncorrect_count_1:{}'.format(count_1))
        print('The number of class not_moving & class moving of training subject {0}: {1},{2}\n'.format(int(num), int(total_0),int(total_1)))
        result_train = result_train.append(
            {'train_subject': int(num), 'correct_not_moving': count_0, 'correct_moving': count_1,
             'total_not_moving': total_0,
             'total_moving': total_1},
            ignore_index=True)


    print("result_train:\n{}".format(result_train))
    print("result:\n{}".format(result))

    # save to csv file
    # csv_name = save_dir_file + "result_of_testing.csv"
    # result.to_csv(csv_name)
    # csv_name_train = save_dir_file + "result_of_training.csv"
    # result_train.to_csv(csv_name_train)

def LOO(group_size, subject_list,batch,epochs,save_dir_file,zero_weight):
    result = pd.DataFrame(columns=["test_subject", "correct_not_moving", "correct_moving", "total_not_moving", "total_moving"])
    for i in range(len(subject_list)):
    # for i in range(1,5):
        test_subject = subject_list[-i]
        print("This is LOO\nThis is to test subject {}".format(test_subject))

        x_train, y_train, x_test, y_test, x_train_dict, y_train_dict = data_load_subjects_LOO(subject_list = subject_list, test_subject=test_subject, group_size=group_size)
        # if i == 0:
        #     #save real value by folders
        #     real_value_path = save_dir_file + "Real_value/"
        #     if not os.path.exists(real_value_path):
        #         m = os.umask(0)
        #         os.makedirs(real_value_path)
        #         os.umask(m)
        #
        #     for keys, values in y_train_dict.items():
        #         y_train_df = pd.DataFrame(values, columns=['Not_moving', 'moving'])
        #         print("\n\nsave the training real value:{}".format("{}_train".format(keys)))
        #         print(real_value_path + "{}.csv".format("{}_train".format(keys)))
        #         y_train_df.append({"Not_moving":np.sum(y_train_df,axis = 0)[0], "moving":np.sum(y_train_df,axis = 0)[1]},ignore_index=True)
        #         y_train_df.to_csv(real_value_path + "Real_{}.csv".format(keys))
        #
        #     y_test_df = pd.DataFrame(y_test,columns = ['Not_moving', 'moving'])
        #     y_test_df.to_csv(real_value_path + "Real_{}.csv".format(test_subject))

        # build mode:
        # zero_weight = zero_weight

        print("/n/nThis is zero_weight of {}/n/n".format(zero_weight))
        model = cnn_model(batch_size=batch, epochs=epochs, class_weight = {0:zero_weight, 1:1}, input_shape=x_train.shape[1:])
        model.fitting(x_train, y_train[:,:-1])
        #save model:
        # model.save_model(name=save_dir_file + "{}_classification.h5".format(YMD))

        result_train = pd.DataFrame(columns = ["train_subject","correct_not_moving","correct_moving","total_not_moving","total_moving"])
        for num in subject_list:
            if num == test_subject:
                print("\nThis is to test subject {}(testing dataset)\n".format(num))
                count_0,count_1,total_0,total_1, accuracy,loss = model.evaluate(x_test, y_test)
                print("(subject_id, accuracy,0_accu, 1_accu):{},{},{},{}".format(int(num), round(accuracy, 2),round(count_0/total_0,2), round(count_1/total_1,2)))
                # predict test dataset
                y_pred = model.predict(x_test)
                print('\ncorrect_count_0\n', count_0, '\ncorrect_count_1\n', count_1)
                print('\nnumber of class 0 & class 1(test)\n', np.sum(y_test, axis=0))
                result = result.append(
                    {"subject": int(num), "correct_not_moving": int(count_0), "correct_moving": int(count_1),
                     "total_not_moving": int(total_0), "total_moving": int(total_1)},
                    ignore_index=True)
            else:
                #predict training dataset
                count_0, count_1, total_0, total_1, accuracy, loss = model.evaluate(x_train_dict[num], y_train_dict[num])
                # print("(subject_id, accuracy,0_accu, 1_accu):{},{},{},{}".format(int(num), round(accuracy, 2),round(count_0/total_0,2), round(count_1/total_1,2)))
                # predict test dataset
                # y_pred = model.predict(x_test)
                result_train = result_train.append(
                    {"subject": int(num), "correct_not_moving": int(count_0), "correct_moving": int(count_1),
                     "total_not_moving": int(total_0), "total_moving": int(total_1)},
                    ignore_index=True)

        print("result_train:\n{}".format(result_train))
        print("result:\n{}".format(result))

        #save to csv file
        # csv_name_train = save_dir_file + "train_result_of_testSub_{}.csv".format(test_subject)
        # result_train.to_csv(csv_name_train)

    #result of testing dataset (save)
    # csv_name = save_dir_file + "result_of_testSub_{}.csv".format(test_subject)
    # result.to_csv(csv_name)

def save_model(group_size, subject_list,batch,epochs,save_dir_file,zero_weight,YMD):
    x_train = []
    y_train = []
    for subject_id in subject_list:
        print("This is load subject {}.\n".format(subject_id))
        path = "/u/erdos/csga/jzhou40/MRI_Project/sub-NC{}_head_pose_data_zhao/".format(
            subject_id)
        # os.listdir: returns a list containing the names of the entries
        label_name = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
        feature_name = [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
        img_dir_name = [f for f in os.listdir(path) if f.endswith('cropped_frames')][0]
        print("img_dir_name", img_dir_name)

        feature_path = path + feature_name
        label_path = path + label_name
        img_path = path + img_dir_name + "/"

        Subject = subject(feature_path=feature_path, label_path=label_path, img_path=img_path, group_size=group_size)
        x, y = Subject.train_data()
        if len(x_train) == 0:
            x_train = x
            y_train = y
        else:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    model = cnn_model(batch_size=batch, epochs=epochs, class_weight={0: zero_weight, 1: 1},
                          input_shape=x_train.shape[1:])
    model.fitting(x_train, y_train[:,:-1])
    # save model:
    model.save_model(name=save_dir_file + "{}_cnn_classification.h5".format(YMD))
    print("\n\nmodel saved\n\n")

######################################
if __name__ == '__main__':
    # for i in range(3):
    for i in range(1):
        currentDT = datetime.datetime.now()
        YMD = "{}{}{}".format(currentDT.year, currentDT.month, currentDT.day)
        HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)

        #############below is parameter###########

        batch = 64
        epochs = 500
        test_sample = 80
        zero_weight = 3
        group_size = 2  # the number of frames as a group to predict observations set size = 1, means (fn - f0) so x.shape = (-1,1,img_h,img_w)//(fn-1 -f0, fn-f0, fn-fn-1)
        #subject_list = [150, 232, 233, 234, 235, 236, 237, 240, 242, 243, 244, 245, 246]
        #subject_list = [232, 233, 234, 235, 236, 237, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255]
        subject_list = [232, 233]
        # save_dir_file = "/u/erdos/students/hyuan11/MRI/upload/cluster/{}_{}_multiSubs_classification/".format(YMD,HMS)
        save_dir_file = "/u/erdos/csga/jzhou40/MRI_Project/ToHealth/"
        ##############above is parameter##########
        # zero_weight=[1,5,10,15,20][i]
        if not os.path.exists(save_dir_file):
            m = os.umask(0)
            os.makedirs(save_dir_file)
            os.umask(m)

        # LOO(group_size, subject_list, batch, epochs, save_dir_file,zero_weight)
        main_allSj(save_dir_file, group_size, subject_list,test_sample,batch,epochs,zero_weight)
        save_model(group_size, subject_list, batch, epochs, save_dir_file, zero_weight,YMD)
