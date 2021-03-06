from __future__ import print_function
import sys
import data_process
import cnn_model
from data_process import *
from cnn_model import *
# import test_video_regression_WithoutGT
import plot
import numpy as np
import pandas as pd
import os
import datetime
import cv2
import time
from keras.utils import plot_model
import shutil
from scipy import stats
import keras.backend as K
from keras.models import load_model


def corr(y):
    y = pd.DataFrame(y[:,:-1], columns=['rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ']).corr(
        method='pearson')
    print('\nmr:the pearson corr for training\n',round(y,2))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def move_file(src_path,des_path,file):
    try:
        f_src = os.path.join(src_path, file)
        f_des = os.path.join(des_path, file)
        shutil.move(f_src,f_des)
    except Exception as e:
        print("mr:move_file error:",e)

def LOOCV(columns_name = None, save_dir_file = None, train_subject_list = None, test_subject_list = None,
                           interval_frame = None, batch = 128, epochs = 500,img_grad = True,combined_image = None):
    print("This is loocv")
    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)

    print("mr:This is multiple subjects")
    print("mr:The columns name are ", columns_name)

    x_train_dic = {}
    y_train_dic = {}
    train_time_dic = {}
    for i in range(len(train_subject_list)):
        print("This is loading subject_{}".format(train_subject_list[i]))
        xTrain, yTrain, time_arr = data_process.data_load_single_subject(columns_name=columns_name,
                                                                            test_subject=train_subject_list[i],
                                                                            interval_frame=interval_frame,
                                                                            Whole_data =True,
                                                                            img_grad = img_grad,
                                                                            combined_image = combined_image)
        print("mr:xTrain: min: {}, max: {}".format(np.min(xTrain),np.max(xTrain)))
        print("mr:training data: subject {}".format(train_subject_list[i]))
        print("mr:xTrain.shape:", xTrain.shape)
        print("mr:yTrain.shape:", yTrain.shape)
        print("mr:have completed loading subject_{} ".format(train_subject_list[i]))
        x_train_dic[train_subject_list[i]] = xTrain
        y_train_dic[train_subject_list[i]] = yTrain
        train_time_dic[train_subject_list[i]] = time_arr

        print("mr:training_sub_{} Max: {}; Min: {}".format(train_subject_list[i],np.max(xTrain), np.min(xTrain)))

    x_val = []
    y_val = []
    x_val_dic = {}
    y_val_dic = {}
    val_time_dic = {}
    for j in range(len(test_subject_list)):
        print("mr:This is loading subject_{}".format(test_subject_list[j]))
        xVal, yVal, time_arr = data_process.data_load_single_subject(
            columns_name=columns_name,
            test_subject=test_subject_list[j],
            interval_frame=False,
            Whole_data=True,
            img_grad = img_grad)
        print("mr:xVal: min: {}, max: {}".format(np.min(xVal), np.max(xVal)))
        print("mr:Validation data: subject {}".format(test_subject_list[j]))
        print("mr:xVal.shape:", xVal.shape)
        print("mr:yVal.shape:", yVal.shape)
        print("mr:have completed loading validation subject_{} ".format(test_subject_list[j]))
        if len(x_val) == 0:
            x_val = xVal
            y_val = yVal
        else:
            x_val = np.concatenate((x_val,xVal),axis=0)
            y_val = np.concatenate((y_val,yVal),axis=0)

        x_val_dic[test_subject_list[j]] = xVal
        y_val_dic[test_subject_list[j]] = yVal
        val_time_dic[test_subject_list[j]] = time_arr
        print("mr:x_val.shape:",x_val.shape)
        print("mr:y_val.shape:",y_val.shape)

    if interval_frame:
        ins_per_sub = len(xTrain) - len(xTrain)%100
    else:
        ins_per_sub = 320

    # self-define model
    # model = cnn_model.get_model_structure(input_shape=x_val.shape[1:], output_column=columns_name)

    # pretrain vgg16
    model = cnn_model.get_model_structure(input_shape=x_val.shape[1:], output_column=columns_name)

    # if columns_name == ['rotX', 'rotY', 'rotZ']:
    #     model = load_model("/u/erdos/students/hyuan11/MRI/model/Dec_15_multi_sub_rot_weights.h5",
    #                        custom_objects={"weighted_MAE":cnn_model.weighted_MAE_rot})
    # elif columns_name == ["transX", "transY", "transZ"]:
    #     model = load_model("/u/erdos/students/hyuan11/MRI/model/Dec_15_multi_sub_rot_weights.h5",
    #                        custom_objects={"weighted_MAE": cnn_model.weighted_MAE_trans})

    model, training_loss, val_loss = cnn_model.model_fitting_generator(model, x_train=x_train_dic, y_train=y_train_dic,
                                                                       x_val=x_val, y_val=y_val,
                                                                       save_dir_file=save_dir_file,
                                                                       batch_size=batch, epochs=epochs,
                                                                       ins_per_sub=ins_per_sub)#,
                                                                      # sub233 = x_train_dic[233])

    #get prediction for models
    for sub in train_subject_list:
        train_pred = model.predict(x_train_dic[sub])
        np.savetxt(save_dir_file + "train_sub_{}_pred.csv".format(sub), np.array(train_pred))
        np.savetxt(save_dir_file + "train_sub_{}_real.csv".format(sub), np.array(y_train_dic[sub]))


    for sub in test_subject_list:
        val_pred = model.predict(x_val_dic[sub])
        np.savetxt(save_dir_file + "val_sub_{}_pred.csv".format(sub), np.array(val_pred))
        np.savetxt(save_dir_file + "val_sub_{}_real.csv".format(sub), np.array(y_val_dic[sub]))

def rotation_multiple_subs(columns_name = None, save_dir_file = None, subject_list = None, test_sample = 100,
                           interval_frame = None, batch = 128, epochs = 500,img_grad = True, data_path = None, combined_image = None):

    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)

    if interval_frame:
        ins_per_sub = interval_frame*220
    else:
        ins_per_sub = 220
    print("This is multiple subjects")
    print("The columns name are ",columns_name)

    x_val = []
    y_val = []
    x_train_dic = {}
    y_train_dic = {}
    x_val_dic = {}
    y_val_dic = {}
    train_time_dic = {}
    val_time_dic = {}
    df_r_p_short = {}
    df_r_p_long = {}

    # data_list = [f for f in os.listdir(data_path) if f[:6] == "sub-NC"]


    for i in range(len(subject_list)):
        # find data by  subject:

        print("\n\nmr:This is loading subject_{}".format(subject_list[i]))
        xTrain, yTrain, xVal, yVal, val_time_arr, train_time_arr = data_process.data_load_single_subject(
                                                                            columns_name = columns_name,
                                                                            test_subject = subject_list[i],
                                                                            test_sample = test_sample,
                                                                            interval_frame = interval_frame,
                                                                            Whole_data = False,
                                                                            img_grad = img_grad,
                                                                            combined_image = combined_image)
        #,                                                                    path = data_path)

        # print("\nmr:\n1st image:\n",xTrain[0,100,:10,:])
        print("training data: subject {}".format(subject_list[i]))
        print("xTrain.shape:", xTrain.shape)
        print("yTrain.shape:", yTrain.shape)
        print("mr:xTrain: min: {}, max: {}".format(np.min(xTrain), np.max(xTrain)))
        print("mr:have completed loading subject_{} ".format(subject_list[i]))

        x_val.append(xVal)
        y_val.append(yVal)
        x_train_dic[subject_list[i]] = xTrain
        y_train_dic[subject_list[i]] = yTrain
        x_val_dic[subject_list[i]] = xVal   #150, 232, 245,  old model constructure,
        y_val_dic[subject_list[i]] = yVal
        train_time_dic[subject_list[i]] = train_time_arr
        val_time_dic[subject_list[i]] = val_time_arr

    x_val = np.array(x_val)
    print("mr:x_val: max: {}; min: {}".format(np.max(x_val),np.min(x_val)))
    x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4])
    print("mr:x_val.shape:",x_val.shape)
    y_val = np.array(y_val)
    y_val = y_val.reshape(-1,y_val.shape[2])
    print("mr:y_val.shape:",y_val.shape)

    model = cnn_model.get_model_structure(input_shape=x_val.shape[1:], output_column = columns_name)
    model, training_loss, val_loss = cnn_model.model_fitting_generator(model, x_train=x_train_dic, y_train=y_train_dic,
                                                             x_val = x_val, y_val = y_val,save_dir_file = save_dir_file,
                                                             batch_size = batch, epochs = epochs,ins_per_sub = ins_per_sub)

    print("mr:start to get prediction and plot")
    for sub, sub_test in x_val_dic.items():
        single_sub_dir = save_dir_file + "{}/".format(sub)
        if not os.path.exists(single_sub_dir):
            m = os.umask(0)
            os.makedirs(single_sub_dir)
            os.umask(m)
        pred_val = model.predict(sub_test)
        pred_train = model.predict(x_train_dic[sub])
        print("pred_val.shape:", pred_val.shape)
        print("pred_train.shape:", pred_train.shape)
        print("val_time_arr.shape", val_time_dic[sub].shape)
        print("train_time_arr.shape", train_time_dic[sub].shape)

        pred_val = pd.DataFrame(np.concatenate((pred_val, val_time_dic[sub]), axis=1),
                                    columns = columns_name + ['time_sec'])
        pred_val.to_csv(single_sub_dir + "Predict_test_{}.csv".format(sub), index=False)

        pred_train = pd.DataFrame(np.concatenate((pred_train, train_time_dic[sub]), axis=1),
                                     columns = columns_name + ['time_sec'])
        pred_train.to_csv(single_sub_dir + "Predict_train_{}.csv".format(sub), index=False)

        real_train = pd.DataFrame(np.concatenate((y_train_dic[sub], train_time_dic[sub]), axis=1),
                                  columns=columns_name + ['time_sec'])
        real_train.to_csv(single_sub_dir + "Label_train_{}.csv".format(sub), index=False)

        real_test = pd.DataFrame(np.concatenate((y_val_dic[sub], val_time_dic[sub]), axis=1),
                                 columns=columns_name + ['time_sec'])
        real_test.to_csv(single_sub_dir + "Label_test_{}.csv".format(sub), index=False)

        try:
            plot.plot_func(columns = columns_name, sub=sub, save_dir=single_sub_dir, label_train=real_train, label_test=real_test,
                           pred_train=pred_train[::interval_frame].reset_index(drop=True), pred_val=pred_val,
                           training_loss=training_loss, val_loss=val_loss)
        except:
            print("Error happens on plotting!")
            pass

        #get testing dataset r2, p_value
        r_list = []
        p_list = []
        r_list_long = []
        p_list_long = []
        for column in columns_name:
            pred_val_short = pred_val[column].values[::39]
            real_val_short = real_test[column].values[1::39]

            pred_val_long = pred_val[column].values
            real_val_long = real_test[column].values[1:]

            min_ = min(len(pred_val_short), len(real_val_short))
            min_long = min(len(pred_val_long), len(real_val_long))

            slope, intercept, r_value, p_value, std_err = stats.linregress(real_test[:min_], pred_val[:min_])
            slope, intercept, r_value_long, p_value_long, std_err = stats.linregress(real_val_long[:min_long],
                                                                                     pred_val_long[:min_long])

        r_list.append(r_value ** 2)
        p_list.append(p_value)
        r_list_long.append(r_value_long ** 2)
        p_list_long.append(p_value_long)
        df_r_p_short[sub] = r_list + p_list
        df_r_p_long[sub] = r_list_long + p_list_long

        tmp = pd.DataFrame(df_r_p_short).transpose()
        tmp.columns = ["rotx_r_2", "rotY_r_2", "rotZ_r_2", "rotX_P", "rotY_P", "rotZ_P"]
        tmp = tmp.sort_index()
        print("\nr2_short", tmp)
        print("saving to r-p file", flush=True)
        tmp.to_csv(save_dir_file + "short_r2_p_value.csv".format)

        tmp_long = pd.DataFrame(df_r_p_long).transpose()
        tmp_long.columns = ["rotx_r_2", "rotY_r_2", "rotZ_r_2", "rotX_P", "rotY_P", "rotZ_P"]
        tmp_long = tmp_long.sort_index()
        print("\nr2_long",tmp_long)
        print("saving to r-p file", flush=True)
        tmp_long.to_csv(save_dir_file + "long_r2_p_value.csv".format)