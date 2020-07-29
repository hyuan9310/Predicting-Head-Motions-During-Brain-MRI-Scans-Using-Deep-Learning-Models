import cv2
import sys
from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import cnn_model
import data_process
import matplotlib.pyplot as plt
import datetime
import time
import plot
from cnn_model import *


def img_read(path):
    """
    read image
    :param path:  image_directory
    :return: image (array like )
    """
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224, 224))
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # sobelx = cv2.equalizeHist(sobelx)
    # sobelx = (sobelx - np.min(sobelx)) / (np.max(sobelx) - np.min(sobelx))
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # sobely = cv2.equalizeHist(sobely)
    # sobely = sobely / np.max(sobely)
    # img = np.concatenate([sobelx,sobely],axis=1)
    img = sobelx
    return img


def plot_func(columns, sub, save_dir, label, pred, name=None,interval_frame = None):
    for column in columns:
        plt.figure(figsize=(20, 12))
        plt.subplot(211)
        plt.plot(np.array(label['time_sec']), np.array(label[column]))
        plt.legend(("ground_truth"))
        plt.title('{} {}'.format(sub, column))
        plt.ylabel(column)
        plt.xlabel('time_sec')
        if columns == ['rotX', 'rotY', 'rotZ']:
            plt.ylim((-0.2, 0.2))
        else:
            plt.ylim((-4, 4))

        plt.subplot(212)
        plt.plot(pred['time_sec'].values[::interval_frame], pred[column].values[::interval_frame])
        plt.title('{} {}'.format(sub, column))
        plt.legend(("prediction"))
        plt.ylabel(column)
        plt.xlabel('time_sec')
        if columns == ['rotX', 'rotY', 'rotZ']:
            plt.ylim((-0.2, 0.2))
        else:
            plt.ylim((-4, 4))

        if name is None:
            plt.savefig(save_dir + '{}_{}.png'.format(sub, column))
            os.chmod(path=save_dir + '{}_{}.png'.format(sub, column),mode=0o777)
            plt.close('all')
        else:
            plt.savefig(save_dir + '{}_{}_{}.png'.format(sub, column,name))
            os.chmod(path=save_dir + '{}_{}_{}.png'.format(sub, column,name), mode=0o777)
            plt.close('all')

def plot_zhao():
    lw = 2.5
    subjects = [250, 251, 252, 253, 254, 255]
    # subjects = [250]
    scans = [1, 2, 3, 4]
    kinds = ["Translation", "Rotation"]

    for subject in subjects:
        for scan in scans:
            fname = 'test_sub_NC{}_0{}.csv'.format(subject, scan)
            data = []
            try:
                with open(fname, 'r') as f:
                    f.readline()
                    for line in f:
                        data.append([float(x) for x in line.split(",")])
            except Exception as e:
                continue

            times = [x[7] for x in data]
            rotx = [x[1] for x in data]
            roty = [x[2] for x in data]
            rotz = [x[3] for x in data]
            transx = [x[4] for x in data]
            transy = [x[5] for x in data]
            transz = [x[6] for x in data]
            for kind in kinds:

                # plt.plot(np.arange(10), np.sin(np.arange(10)))
                fig = plt.figure(figsize=(20, 6))
                # ax.plot(times[1:-1:43], rotx[1:-1:43])
                # ax.plot(times[1:-1:43], roty[1:-1:43])
                # ax.plot(times[1:-1:43], rotz[1:-1:43])
                ax = fig.add_axes([0.05, 0.1, 0.83, 0.75])
                if (kind == "Translation"):
                    x = ax.plot(times[1:-1:43], transx[1:-1:43], label="Trans_X", linewidth=lw)
                    y = ax.plot(times[1:-1:43], transy[1:-1:43], label="Trans_Y", linewidth=lw)
                    z = ax.plot(times[1:-1:43], transz[1:-1:43], label="Trans_Z", linewidth=lw)
                    ax.set(xlim=(-5, max(times) + 5), ylim=(-8.0, 8.0))
                else:
                    x = ax.plot(times[1:-1:43], rotx[1:-1:43], label="Rot_X", linewidth=lw)
                    y = ax.plot(times[1:-1:43], roty[1:-1:43], label="Rot_Y", linewidth=lw)
                    z = ax.plot(times[1:-1:43], rotz[1:-1:43], label="Rot_Z", linewidth=lw)
                    ax.set(xlim=(-5, max(times) + 5), ylim=(-0.2, 0.2))

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(14)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)
                ax.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0., fontsize=16)
                ax.set_title('Subject {}_0{} {} Predictions'.format(subject, scan, kind), fontsize=24)
                plt.savefig('test_sub_NC{}_0{}_{}.png'.format(subject, scan, kind))

def test_single_subject(model = None,columns_name=None,subject=None, save_dir_file=None,
                        interval_frame = None, img_grad = None, Whole_data=None):
    interval_frame = interval_frame
    Whole_data = Whole_data
    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)

    print("This is multiple subjects")
    print("The columns name are ", columns_name)
    print("interval_frame: {};\nWhole_data: {}".format(interval_frame,Whole_data))

    print("This is loading subject_{}".format(subject))
    if Whole_data:
        print("Whole data: ",Whole_data)
        xTrain, yTrain, time_arr = data_process.data_load_single_subject(columns_name=columns_name,
                                                                        test_subject=subject,
                                                                        test_sample = 100,
                                                                        interval_frame=interval_frame,
                                                                        Whole_data =Whole_data,
                                                                         img_grad = img_grad)

    else:
        print("5 mins train, 2 mins test")
        xTrain, yTrain, xTest, yTest, test_time_arr, train_time_arr = data_process.data_load_single_subject(
            columns_name=columns_name,
            test_subject=subject,
            test_sample = 100,
            interval_frame=interval_frame,
            Whole_data=Whole_data,
            img_grad = img_grad)

        print("xTest: min: {}, max: {}".format(np.min(xTest), np.max(xTest)))

    xTrain = (xTrain - np.min(xTrain))/(np.max(xTrain)- np.min(xTrain))
    # print("\n1st image:\n", xTrain[0, 100, :10, :])
    print("xTrain: min: {}, max: {}".format(np.min(xTrain),np.max(xTrain)))
    print("xTrain.shape:", xTrain.shape)
    print("yTrain.shape:", yTrain.shape)
    print("have completed loading subject_{} ".format(subject))
    print("training_sub_{} Max: {}; Min: {}".format(subject,np.max(xTrain), np.min(xTrain)))
    # load model
    # use GPU
    gpu_no = 0
    if Whole_data:
        with tf.device("/gpu:" + str(gpu_no)):
            print("this is to run gpu")
            y_pred = model.predict(xTrain)
            print("y_pred.shape:", y_pred.shape)
        pred_train = pd.DataFrame(np.concatenate((y_pred, train_time_arr), axis=1),
                                  columns=columns_name + ['time_sec'])
        pred_train.to_csv(save_dir_file + "train_sub_{}.csv".format(subject))
        plot_func(columns_name, subject, save_dir_file, yTrain, pred_train, name="train")

    else:
        with tf.device("/gpu:" + str(gpu_no)):
            print("this is to run gpu")
            y_pred_train = model.predict(xTrain)
            print("y_pred_train.shape:", y_pred_train.shape)
            y_pred_test = model.predict(xTest)
            print("y_pred_test.shape:", y_pred_test.shape)

        pred_train = pd.DataFrame(np.concatenate((y_pred_train, train_time_arr), axis=1),
                                  columns=columns_name + ['time_sec'])
        pred_train.to_csv(save_dir_file + "train_sub_{}.csv".format(subject))
        print("pred_train.shape:",pred_train.shape)

        pred_test = pd.DataFrame(np.concatenate((y_pred_test, test_time_arr), axis=1),
                                  columns=columns_name + ['time_sec'])
        pred_test.to_csv(save_dir_file + "test_sub_{}.csv".format(subject))
        print("pred_test.shape:", pred_test.shape)

        plot_func(columns_name, subject, save_dir_file, yTrain, y_pred_train, name="train")
        plot_func(columns_name, subject, save_dir_file, yTest, y_pred_test, name = "test")



################# main ##################
# if __name__ =="__main__":
def  main_test_NoGT(judge_rot_trans,withGT = True,interval_frame = None, img_grad = None,Whole_data = False,
                    model_location =None,save_path = None):
    """
    sys.argv[1]:model's name (eg: Dec_15_multi_sub_rot_weights.h5) : str
    sys.argv[2]:subject_name: str (eg:"NC250_01")
        ['NC254_02', 'NC250_03', 'NC250_04', 'NC250_02', 'NC254_04', 'NC254_03', 'NC251_02', 'NC255_04', 'NC255_03',
         'NC255_02', 'NC251_03', 'NC252_01', 'NC254_01', 'NC250_01', 'NC251_01', 'NC255_01', 'NC252_02']
    sys.argv[3]: "rot" or "trans": str

    """
    model_location = model_location
    judge_rot_trans = judge_rot_trans


    Whole_data = Whole_data
    img_grad = img_grad
    interval_frame = interval_frame

    print("launching test_video_regression_WithoutGT.py!")
    currentDT = datetime.datetime.now()
    MD = "{}{}".format(currentDT.month, currentDT.day)
    HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)
    start = time.time()
    print("This is to predict {}".format(judge_rot_trans))
    print("with label:",withGT)

    # /u/erdos/students/hyuan11/MRI/upload
    # new_dir = "/home/huiyuan/summer_project/code/{0}_{1}_test_new_sub_{2}/".format(MD,HMS,judge_rot_trans)
    new_dir = save_path + "/{0}_{1}_test_new_sub_{2}/".format(MD,HMS,judge_rot_trans)
    if not os.path.exists(new_dir):
        m = os.umask(0)
        os.makedirs(new_dir)
        os.umask(m)

    # test_single_subject(subject=sys.argv[2], columns_name=["rotX", "rotY", "rotZ"], save_dir_file="/u/erdos/students/hyuan11/MRI/{}_{}_{}")
        try:
            if judge_rot_trans == 'rot':
                print("(main)columns: rotX , rotY, rotZ")
                model = load_model(model_location,
                                   # custom_objects={'r_squared': cnn_model.r_squared,
                                  custom_objects={'weighted_MAE_rot': cnn_model.weighted_MSE_rot})

                print("{} {}".format(MD,HMS),"model_location:",model_location)
            else:
                print("(main)columns: transX , transY, transZ")
                model = load_model(model_location,
                                   custom_objects={'weighted_MAE_trans': cnn_model.weighted_MAE_trans})
            print("have loaded model!")

        except Exception as e:
            print("load model error")
            print("error :",e)


    # for subject in [248,249,250,251,252,254,255]:
    # for subject in [150, 232, 245, 233, 234, 235, 236, 237, 240, 242, 243, 244, 246]:
    # for subject in [150]:
    for subject in ['NC254_02', 'NC250_03', 'NC250_04', 'NC250_02', 'NC254_04', 'NC254_03', 'NC251_02', 'NC255_04',
                    'NC255_03',
                    'NC255_02', 'NC251_03', 'NC252_01', 'NC254_01', 'NC250_01', 'NC251_01', 'NC255_01', 'NC252_02']:
        try:
            if judge_rot_trans == 'rot':
                test_single_subject(model=model, columns_name=["rotX", "rotY", "rotZ"], subject=subject,
                                    save_dir_file=new_dir + "sub_{0}/".format(subject),
                                    interval_frame = interval_frame, img_grad = img_grad,Whole_data=Whole_data)

            if judge_rot_trans == 'trans':
                test_single_subject(model=model, columns_name=["transX", "transY", "transZ"],
                                                     subject=subject,
                                                     save_dir_file=new_dir + "sub_{0}/".format(subject),
                                    interval_frame = interval_frame, img_grad = img_grad,Whole_data = Whole_data)

        except Exception as e:
            print("error happened in {}".format(subject))
            print("error: ", e)
            pass
        
    end = time.time()
    print("time_cost:",end-start)