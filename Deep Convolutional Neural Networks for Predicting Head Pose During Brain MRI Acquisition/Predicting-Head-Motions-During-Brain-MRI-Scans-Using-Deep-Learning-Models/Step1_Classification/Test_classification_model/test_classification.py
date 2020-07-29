from __future__ import print_function
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import cv2
import os
import sys
import warnings
import overlay
import time

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import to_categorical

####################################self_defined function##################################

class subject(object):
    def __init__(self,head_pose, frame_times, img_path, group_size):
        self.head_pose = head_pose  #six value of headpose
        self.frame_times = frame_times      #frame names
        self.img_path = img_path          #directory of frames
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
        # assert sub in [150, 232, 233, 234, 235, 236, 237, 240, 242, 243, 244, 245, 246]:
        # path = "/u/erdos/students/hyuan11/MRI/gpfs/data/epilepsy/mri/hpardoe/head_pose_datasets/sub-NC{}_head_pose_data_zhao/".format(sub)
        data = pd.read_csv(self.head_pose, delimiter=',')
        pic_name = pd.read_csv(self.frame_times, delimiter=',')
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

        label = np.array(data.iloc[:, :-1])
        time = np.array(data["time_sec"],dtype=float)
        # assume the first row is initial frame, the time at that frame may not be 0
        Y = label - label[0]

        y = [0]
        for i in range(1, len(Y)):
            if np.any(np.absolute(Y[i, :3]) >= 0.025):  # 1mm == 1
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
            pic_list.append(img)
        X = np.array(pic_list)
        return X, Y

    def batch_feature(self, x, y, group_size):
        """
        input:  x: array(frames,img_row,img_col,size)
                y: array(frames,1)

        output: x_batch:(None,img_row, img_col,size)
                y_batch:(None,1)
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
            feature = np.array(feature).reshape(x.shape[1], x.shape[2], -1)
            # create one batch as one input feature, shape = [256,256,3*(size-1)]
            x_batch.append(feature)
        return np.array(x_batch).reshape(len(x_batch), x.shape[1], x.shape[2],-1), np.array(y_batch).reshape(-1, 2)

    def train_data(self):
        # data preprocessing
        x, y  = self.data_loading()
        x = x / 255
        # batch_feature:
        x, y = self.batch_feature(x, y, group_size=self.group_size)

        # to_categorical:
        # y = to_categorical(y[:,:-1], 2)
        print("\n\nthe length of x and y after batching:\n{}\n{}\n\n".format(np.shape(x),np.shape(y)))
        return x, y

def data_preprocessing(path):
    #the path to the directory of data
    #eg:sub-NC150_head_pose_data_zhao  (in this directory, it includes headpose.csv,frame_time.csv & the directory of frames
    head_pose = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
    frame_times = [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
    img_dir_name = [f for f in os.listdir(path) if f.endswith('cropped_frames')][0]
    # print("img_dir_name", img_dir_name)

    head_pose = path + head_pose
    frame_times = path + frame_times
    img_path = path + img_dir_name + "/"
    # print("img_path: ", img_path)
    subject_test = subject(head_pose=head_pose, frame_times=frame_times, img_path=img_path, group_size=2)

    x_test, y_test = subject_test.train_data()
    return  x_test, y_test

def Real_value(real_file):
    data = pd.read_csv(real_file)
    sixValue = np.array(data.iloc[:,:6])
    y = []
    for i in range(len(data)):
        if np.any(np.absolute(data.iloc[i, :3]) >= 0.025) or np.any(np.absolute(data.iloc[i, 3:-1]) >= 1):  # 1mm == 1
            y.append("Moving")
        else:
            y.append("Not_moving")
    time = np.array(data.loc[:,'time_sec'],dtype=float)
    newDF = pd.DataFrame({"RotX":sixValue[:,0],'RotY':sixValue[:,1],'RotZ':sixValue[:,2],
                          'TransX':sixValue[:,3],'TransY':sixValue[:,4],'TransZ':sixValue[:,5],
                          'Label':y,'time_sec':time})
    # print("newDF:\n",newDF.head())
    return newDF


def img_read(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)/255
    return img

def input_norm(img):
    img = img/255
    return img

def test_without_label(path):
    frame_times = path + [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
    img_name = path + [f for f in os.listdir(path) if f.endswith('cropped_frames')][0] + '/'

    pic_name = pd.read_csv(frame_times, delimiter=',')
    pic_name["time_sec"] = round(pic_name["time_sec"], 2)  # match label_time_sec
    pic_name = pic_name[pic_name["time_sec"]>=0].reset_index(drop=True)
    pic_name["file"] = list(map(lambda x: x.split('/')[-1], pic_name["file"]))  # modify img_name
    print("pic_name:\n",pic_name.head(5))
    frame_0 = img_read(img_name+pic_name.loc[0,"file"])
    # print("pic_name.iloc[39,:]:\n",pic_name.iloc[39,:])
    pic_list = []
    time_list = []
    name_list = []
    for i in range(1, len(pic_name)-1):
    #for i in range(1,400):
        print("load the No.{} image data:\n".format(i))
        # adding fn-1 - f_0
        f_head = img_read(img_name+pic_name.loc[i,"file"]) - frame_0
        # print("each head_time:",pic_name.loc[i,"time_sec"])
        # adding fn - f_0
        f_end = img_read(img_name+pic_name.loc[i+1,"file"]) - frame_0
        # print("each tail_time:",pic_name.loc[i+39, "time_sec"])
        # adding fn-fn-1
        f_diff = f_end - f_head
        batch_x = np.array([f_head,f_end,f_diff]).reshape(frame_0.shape[0],frame_0.shape[1],-1)
        # save each group into list
        pic_list.append(batch_x)
        time_list.append(pic_name.loc[i+1,"time_sec"])
        name_list.append(pic_name.loc[i+1,'file'])
    
    X = np.array(pic_list)
    print("load feature done!")
    func_Time = np.array(time_list)
    func_Name = np.array(name_list)
    return X,func_Time.reshape(len(func_Time),1),func_Name.reshape(len(func_Name),1)

# def func_test_classification():
if __name__=='__main__':
    start_time = time.time()
    warnings.filterwarnings("ignore")
    path = sys.argv[1]
    video_name = sys.argv[2]
    data_with_ground_truth = False  #boolean

    if not path[-1]=='/':
        path = path + '/'

    #create folder based on argv[1]
    name = path.split('/')[-2]
    print("path:\n",path)
    print("\nsave_name:",name)

    if not os.path.exists(os.getcwd() + "/output/" + "{}/".format(name)):
        m = os.umask(0)
        os.makedirs(os.getcwd() + "/output/" + "{}/".format(name))
        os.umask(m)
    save_path = os.getcwd() + "/output/" + "{}/".format(name)
    print("save_directory_name: {}".format(save_path))

    if data_with_ground_truth:
        x_test, y_test = data_preprocessing(path = path)
        groud_truth = np.array(["Moving" if x[0] >= 0.5 else "Not_moving" for x in y_test]).reshape(-1,1)
        # print(x_test.shape,y_test.shape)
        model = load_model("./20191027_cnn_classification.h5")

        # loss, accuracy = model.evaluate(x_test, y_test[:,:-1])
        total_0 = len(y_test) - np.sum(y_test[:,:-1])
        total_1 = np.sum(y_test[:,:-1])

        y_pred = model.predict(x_test)
        y_pred_csv = ["_Moving" if x[0] >= 0.5 else "Not_moving" for x in y_pred]
        y_pred_csv = pd.DataFrame(np.concatenate((np.array(y_pred_csv).reshape(len(y_pred_csv),1),
                                                  y_pred.reshape(len(y_pred_csv),1),
                                                  groud_truth,
                                                  y_test[:,-1].reshape(len(y_test),-1)),
                                                 axis =1),
                                columns=["Pred_label","prediction","ground_truth","time_sec"])
        y_pred_csv.to_csv(save_path + "prediction_subject.csv")

        #evaluate
        y_pred_eva = list(map(lambda x: 1 if x[0] >= 0.5 else 0, y_pred))
        y_pred_eva = np.array(y_pred_eva)
        count_1 = 0
        count_0 = 0
        for i in range(len(y_test)):
            if y_pred_eva[i] == 1 and y_test[i] == 1:
                count_1 += 1
            if y_pred_eva[i] == 0 and y_test[i] == 0:
                count_0 += 1

        print("(overall_accuracy,Not_moving_accu, Moving_accu): {},{},{}".format(round((count_0 + count_1)/(total_0 + total_1), 2),
                                                                                   round(count_0 / total_0, 2),
                                                                                   round(count_1 / total_1, 2)))
        print("total number of not_moving: {}".format(total_0))
        print("Correct number of not-moving: {}".format(count_0))
        print("total number of moving: {}".format(total_1))
        print("Correct number of moving: {}".format(count_1))

        head_pose = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
        real_df = Real_value(real_file=path + head_pose)
        real_df["time_sec"] = real_df['time_sec'].astype(float)
        y_pred_csv["time_sec"] = y_pred_csv["time_sec"].astype(float)
        info = pd.merge(real_df, y_pred_csv, how='left', on='time_sec')
        info.to_csv(save_path + "{}.csv".format(video_name))
        print("prediction csv file is done")
        overlay.Display(subject_path=path, pred_file=save_path + "prediction_subject.csv", real_file=path + head_pose,
                        save_path=save_path, video_name=video_name, label=data_with_ground_truth)

    else:
        # x_test,Time, Name= test_without_label(path=path)
        # time_read = list(map(lambda x:"{} mins {} sec".format(int(x[0])//60,round(x[0]%60,3)),Time))
        # print("input_feature_shape:",x_test.shape)
        # model = load_model("./20191027_cnn_classification.h5")
        # print("predicting ...")
        # y_pred = model.predict(x_test)
        # y_pred_csv = ["Moving" if x[0] >= 0.5 else "Not_moving" for x in y_pred]
        # y_pred_csv = pd.DataFrame(np.concatenate((Name, np.array(y_pred_csv).reshape(-1, 1),Time,np.array(time_read).reshape(-1,1))
        #                                          ,axis=1),
        #                           columns=["Name","Pred_label","time_sec","time"])
        # y_pred_csv.to_csv(save_path + "prediction_subject.csv")
        overlay.Display(subject_path=path, pred_file=save_path + "prediction_subject.csv", real_file=None,
                        save_path=save_path, video_name=video_name, label=data_with_ground_truth)

            # print("The next step is to do overlay for the video")

    end_time = time.time()

    print("\ntotal testing time:{}".format(end_time-start_time))
