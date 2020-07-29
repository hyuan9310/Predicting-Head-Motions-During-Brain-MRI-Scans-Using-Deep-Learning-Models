import matplotlib
matplotlib.use('Agg')
import cv2
import sys
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from scipy import stats


def crop_img(img):
    x,y = img.shape;
    x = int(x*2/3)
    y = int(y/2)
    img = img[:x, :y]
    return img

def img_read(path,img_grad,crop, combined_image):
    # try:
    if not img_grad:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        img = np.squeeze(img,axis=2)
    else:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (224, 224))
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        if combined_image:
            img = np.array([sobelx,img])
        else:
            img = sobelx

    if crop:
        img = crop_img(img)
    return img
    # except Exception as e:
    #     print(path[-10:])
    #     print(e)

def plot_func(columns, sub, save_dir, label, pred, name=None, withGT = True):
    for column in columns:
        if withGT:
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
            # plt.figure(figsize=(20, 12))
            plt.plot(np.array(pred['time_sec']), np.array(pred[column]))
            plt.title('{} {}'.format(sub, column))
            plt.legend(("prediction"))
            plt.ylabel(column)
            plt.xlabel('time_sec')
            if columns == ['rotX', 'rotY', 'rotZ']:
                plt.ylim((-0.2, 0.2))
            else:
                plt.ylim((-4, 4))
        else:
            plt.figure(figsize=(20,12))
            plt.plot(np.array(pred['time_sec']), np.array(pred[column]))
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


def load_data(store_image_dir,interval_frame,img_grad,crop, combined_image):
    x_input = []
    img_dir = store_image_dir
    frame_0 = img_read(img_dir+"frame_000001.png",img_grad,crop, combined_image)
    print("os.listdir(img_dir):",os.listdir(img_dir)[:5])
    sorted_img_list = sorted(os.listdir(img_dir),key=lambda x: x[-10:-4])
    print("sorted_img_list:",sorted_img_list[:5])
    #predict per 1.3s:
    # for img_path in sorted_img_list[39::39]:

    #predict all frames:
    if interval_frame:
        interval_frame_num = interval_frame
        length = int(39 // interval_frame_num)
        print("intermediate number:",interval_frame_num - 1)
        x_train_list = []

        for i in range(39, len(sorted_img_list)-1,39):
#        for i in range(39,391,39):  # only for test
            if i == 39:
                print("first predict image:",sorted_img_list[i])
            img_num = int(sorted_img_list[i][6:12])

            name_head = img_dir + 'frame_' + str(img_num - length).zfill(6) + '.png'
            f_head = img_read(name_head,img_grad,crop, combined_image) - frame_0

            name_tail = img_dir + 'frame_' + str(img_num).zfill(6) + '.png'
            f_tail = img_read(name_tail,img_grad,crop, combined_image) - frame_0

            f_diff = f_tail - f_head

            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            for j in range(interval_frame_num-1):
                try:
                    name_head = img_dir + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
                    # print("head:", name_head)

                    f_head = img_read(name_head,img_grad,crop, combined_image) - frame_0

                    name_tail = img_dir + 'frame_' + str(img_num + length * (j+1)).zfill(6) + '.png'
                    f_tail = img_read(name_tail,img_grad,crop, combined_image) - frame_0
                except:
                    # print("error: ", 'frame_' + str(img_num + length * (j+1)).zfill(6) + '.png')
                    pass

                f_diff = f_tail - f_head

                one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
                x_train_list.append(one_input_img)

    #
    # for img_path in sorted_img_list[39::]:
    # # for img_path in sorted_img_list[1:20]:  #only test 10 frames for test!!!!
    #     # print("img_path:",img_path)
    #
    #     #predict only 1.3s frames
    #     # f_head = img_read(img_dir + "frame_" + str(int(img_path[-10:-4]) - 39).zfill(6) + ".png") - frame_0
    #     #all frame predictions
    #     f_head = img_read(img_dir + "frame_" + str(int(img_path[-10:-4]) - 1).zfill(6) + ".png") - frame_0
    #
    #     f_tail = img_read(img_dir +  img_path) - frame_0
    #     f_diff = f_tail - f_head
    #     x_input.append([f_head, f_tail, f_diff])

    x_input = np.array(x_train_list)
    x = np.transpose(x_input,[0,2,3,1])
    print("x.shape", x.shape)
    print("load input done!!")

    return x

def test_single_subject_gpu(data_path = None, model_name = None, subject=None,interval_frame = None,
                            columns_name=["rotX", "rotY", "rotZ"], save_dir_file=None,use_gpu=True,
                            img_grad = None, withGT = None,crop = False, combined_image= False):
    print("test column name:", columns_name)
    print("the name of directory to save img:", save_dir_file)
    if save_dir_file[-1] != "/":
        save_dir_file = save_dir_file + '/'

    # check if directory exist:
    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)

    if withGT:
 
        print("data_path", data_path)
         
        try:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frame")][0] + "/"
        except:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frames")][0] + "/"
        
        print("store_image_dir",store_image_dir)
        real_path  = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("head_pose.csv")][0]
        df_real = pd.read_csv(real_path,delimiter=",")
        df_real.to_csv(save_dir_file + "{}_rot_real_short.csv".format(subject))  # eg: ./NC250_01/NC250_01_rot.csv


        print("df_real.shape",df_real.values.shape)
        print("df_real:\n",df_real.head(3))
    else:
        try:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frame")][0] + "/"
        except:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frames")][0] + "/"

    start_image = 0
    filename = data_path + [f for f in os.listdir(data_path) if f.endswith('frame_times.csv')][0]

    with open(filename) as f:
        f.readline()
    #    print(filename)
        for line in f:
    #        print(line.split(",")[-1])
            if float(line.split(",")[-1]) < 0:
                start_image += 1
            else:
                break

    print('start_image: {}'.format(start_image))
    # load data
    x_input = load_data(store_image_dir,img_grad = img_grad, interval_frame = interval_frame,crop = crop, combined_image= combined_image)
    x_input = x_input[start_image:]
    print("x_input.shape",x_input.shape)

    #perdict all:
    time_all = np.arange(start=0,stop=len(x_input)) * 0.033333*(39//interval_frame) +1.3
    time_all = list(map(lambda x: round(x,2),time_all))
    time_all = np.array(time_all)
    time_all = time_all.reshape(-1,1)
    # print("time_all:{}\nlength:{}\n".format(time_all[:5],len(time_all)))

    # load model
    if use_gpu:
        # use GPU
        gpu_no = 0
        with tf.device("/gpu:" + str(gpu_no)):
            print("this is to run gpu")
            model = model_name
            y_pred = model.predict(x_input)
            # K.clear_session()
    else:
        model = model_name
        y_pred = model.predict(x_input)
        # K.clear_session()

    print("y_pred.shape:",y_pred.shape)
    pred_train = pd.DataFrame(np.concatenate((y_pred, time_all), axis=1),
                              columns = columns_name + ['time_sec'])
    # to CSV file: note exlude time column
    if columns_name == ["rotX", "rotY", "rotZ"]:
        pred_train.to_csv(save_dir_file + "{}_rot_pred_long.csv".format(subject)) #eg: ./NC250_01/NC250_01_rot.csv
    else:
        pred_train.to_csv(save_dir_file + "{}_trans_pred_long.csv".format(subject))

    if columns_name == ["rotX", "rotY", "rotZ"]:
        pred_train_short = pred_train.iloc[::39,:]
        pred_train_short.to_csv(save_dir_file + "{}_rot_pred_short.csv".format(subject))
        new_real = {}
        for column in ["rotX", "rotY", "rotZ","time_sec"]:
            new_real[column] = np.interp(pred_train['time_sec'].values, df_real['time_sec'].values, df_real[column].values)
        df_new_real = pd.DataFrame(new_real)
        df_new_real.to_csv(save_dir_file + "/{}_rot_real_long.csv".format(sub))

    # print("225 withGT:",withGT)
    if withGT:
        # print("enter with Ground truth")
        plot_func(columns = columns_name, sub = subject, save_dir = save_dir_file, label = df_real, pred = pred_train,
                  withGT = withGT)
    else:
        print("enter No Ground truth")
        plot_func(columns=columns_name, sub=subject, save_dir=save_dir_file, label=None, pred = pred_train,
                  withGT = withGT)

    if withGT:
        return pred_train, df_real, df_new_real
    else:
        return pred_train


def weighted_MAE_rot(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.abs((y_pred - y_true)*y_weight))  #modify： if <10^-4, *1; else 100  (233,234,244,245)

def weighted_MAE_trans(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*100*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.abs((y_pred - y_true)*y_weight))

def weighted_MSE_rot(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)

    # ### New weight!!!code report error
    # y_weight = tf.where(condition,
    #                     tf.greater(tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32")),
    #                     tf.ones_like(y_true,dtype="float32"))

    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*100*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)

################# main #################################################################################################

def main_test_NoGT(interval_frame, img_grad, withGT, data_path, save_path,
                   judge_rot_trans,model,crop,sub, r2_with_all_data, combined_image):
    currentDT = datetime.datetime.now()
    YMD = "{}{}{}".format(currentDT.year, currentDT.month, currentDT.day)
    HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)
    start = time.time()

    # choose to use gpu or cpu
    use_gpu = True
    if data_path[-4:] == ".mp4":
        VIDEO_FILE = data_path  # eg:./gftp/headpose/NC250_01/NC250_01.mp4
        if not os.path.exists("./{}/".format(VIDEO_FILE.split("/")[-1].rstrip()[:-4])):
            m = os.umask(0)
            os.makedirs("./{}/".format(VIDEO_FILE.split("/")[-1].rstrip()[:-4]))
            os.umask(m)
        save_dir_file = "./{}/{}".format(VIDEO_FILE.split("/")[-1].rstrip()[:-4],
                                         VIDEO_FILE.split("/")[-1].rstrip()[:-4])  # eg: ./NC250_01/NC250_01
        print("save_dir_file:./{}\n".format(save_dir_file.split('/')[1]))

        # cropped frames
        folder_name = VIDEO_FILE.split("/")[-1].rstrip()[:-4] + "/cropped_frame/"  # eg: NC250_01/frames/
        if os.path.exists("./{}".format(folder_name)) and len(
                os.listdir("./{}".format(folder_name))) > 10:  # check if cropped already and existing images
            print("Warning: This video has already been cropped. Hence, the program will use the existing images")
            pass
        elif os.path.exists("./{}".format(folder_name)) and len(os.listdir("./{}".format(folder_name))) < 10:
            print("existing directory for image cropped but it's null!")
            os.system("rm -r ./{}".format(folder_name))
            print("Cropping the video to images!")
            os.makedirs('./{}'.format(folder_name))
            os.system("ffmpeg -i {} ./{}/frame_%06d.png".format(VIDEO_FILE, folder_name))
            print("\n Cropping is done!")

        else:
            print("Cropping the video to images!")
            os.makedirs('./{}'.format(folder_name))
            os.system("ffmpeg -i {} ./{}/frame_%06d.png".format(VIDEO_FILE, folder_name))
            print("\n Cropping is done!")
    else:
        dir_list = [f for f in os.listdir(data_path)]
        save_dir_file = save_path
        if not os.path.exists(save_dir_file):
            m = os.umask(0)
            os.makedirs(save_dir_file)
            os.umask(m)

    # print("load model for columns: rotX , rotY, rotZ")
    # try:
    #     model_rot = load_model(model_location, custom_objects={'weighted_MSE_rot': weighted_MSE_rot})
    # except:
    #     model_rot = load_model(model_location, custom_objects={'weighted_MAE': weighted_MSE_rot})
    # print("load model for columns: transX , transY, transZ")
    # model_trans = load_model("./Dec_20_multi_sub_trans_weights.h5",
    #                        custom_objects={'weighted_MAE_trans': weighted_MAE_trans})
    # print("have loaded model!")

    if judge_rot_trans == "rot":
        columns_name = ["rotX", "rotY", "rotZ"]
    else:
        columns_name = ["transX", "transY", "transZ"]

    # use gpu
    print("Using GPU !")
    if withGT:
        pred_rot, df_real, df_new_real = test_single_subject_gpu(data_path=data_path, model_name=model,
                                                       subject=sub,columns_name=columns_name,
                                                       save_dir_file=save_dir_file, use_gpu=use_gpu,
                                                       img_grad=img_grad, interval_frame=interval_frame,
                                                       withGT=withGT, crop=crop, combined_image = combined_image)

        # calc pvalue and r squared
        r_list = []
        p_list = []
        for column in columns_name:
            if column == "time_sec":
                pass
            else:
                if not r2_with_all_data:
                    pred_val = pred_rot[column].values[::39]
                    real_val = df_real[column].values[1:]
                    print("r2_p: \nprediction:\n{}\nreal:\n{}".format(pred_val.head(3),real_val.head(3)))
                    min_ = min(len(pred_val),len(real_val))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(real_val[:min_],pred_val[:min_])
                    r_list.append(r_value ** 2)
                    p_list.append(p_value)
                else:
                    pred_val = pred_rot[column].values
                    real_val = df_new_real[column].values[1:]
                    min_ = min(len(pred_val), len(real_val))
                    print("r2_p: \nprediction:\n{}\nreal:\n{}".format(pred_val.head(3),real_val.head(3)))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(real_val[:min_], pred_val[:min_])
                    r_list.append(r_value ** 2)
                    p_list.append(p_value)
        return r_list,p_list

    else:
        pred_rot = test_single_subject_gpu(data_path=data_path, model_name=model_rot,subject=sub,
                                           columns_name=columns_name, save_dir_file=save_dir_file, use_gpu=use_gpu,
                                           img_grad=img_grad, interval_frame=interval_frame, withGT=withGT, crop=crop,
                                           combined_image = combined_image)
        print("completed rotation part")


    # # use gpu
    # pred_trans = test_single_subject_gpu(store_image_dir = "./{}".format(folder_name), model_name = model_trans,
    #                                          subject="{}".format(VIDEO_FILE.split("/")[-1].rstrip()[:-4]),
    #                                          columns_name=["transX", "transY", "transZ"], save_dir_file=save_dir_file,use_gpu=use_gpu)
    # print("completed translation part")

    # combined rotation and translation
    # result = pd.concat([pred_rot.iloc[:,:-1],pred_trans],axis = 1)
    # result.to_csv("./{}/rot_trans.csv".format(YMD,HMS))
    end = time.time()
    print("time_cost:", end - start)




if __name__ == "__main__":
    ###############################################  parameters ###########################
    if sys.argv[1] == "noskip":
        interval_frame = 39  # True or False
    elif sys.argv[1] == "skip":
        interval_frame = 1
    else:
        assert (int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 39), "Please enter a number larger than 1 and smaller than 39"
        interval_frame = int(sys.argv[1])
    print("intermeidate:",sys.argv[1])

    if sys.argv[2] == "original":
        img_grad = False  # True or False
    elif sys.argv[2] == "gradient":
        img_grad = True
    print("image loading:", sys.argv[2])

    # withGT = sys.argv[3] # true or false
    # if withGT == "True" or withGT == "true":
    #     withGT = True
    # else:
    #     withGT = False
    # print("withGT:", withGT)

    data_path = sys.argv[3] # load data
    print("data_path:",data_path)

    save_path = sys.argv[4]  # save results
    if save_path[-1] != "/":
        save_path = save_path + "/"
    print("save_path:", save_path)

    try:
        judge_rot_trans = sys.argv[5]
    except:
        judge_rot_trans = "rot"
        # judge_rot_trans = 'rot'  # rot or trans
    print("rot or trans:",judge_rot_trans)

    try:
        model_location = sys.argv[6]
        # model_location = "/home/huiyuan/summer_project/code/202035_175349_multiSubs_rot/weights_single_best.h5"
    except:
        pass
    print("model location:", sys.argv[6])

    crop_0 = sys.argv[7]
    print("crop:",crop_0)
    if crop_0 == "full":
        crop = False
    elif crop_0 == "quarter":
        crop = True

    r2_size = sys.argv[8]
    if r2_size == "all":
        r2_with_all_data = True
    elif r2_size == "part":
        r2_with_all_data = False

    if sys.argv[9] == "combined":   # True or False
        combined_image = True
    elif sys.argv[9] == "single":
        combined_image = False

    # test_list_withGT = [247, 248, 249, 250, 251, 252, 254, 255, 150, 232, 233, 234, 235, 236, 237, 240, 242, 243, 244, 245, 246]
    test_list_withGT = [240, 242]#, 243, 244, 245, 246]
    #test_list_withGT = [246]
    test_list_noGT = []

    df_r_p = {}
    ################################################################################################################
    # load model:
    print("load model for columns: rotX , rotY, rotZ")
    if judge_rot_trans == "rot":
        try:
            model_rot = load_model(model_location, custom_objects={'weighted_MSE_rot': weighted_MSE_rot})
        except:
            model_rot = load_model(model_location, custom_objects={'weighted_MAE': weighted_MSE_rot})
    print("have loaded model!")

    if data_path[-3:] != "mp4":
        if data_path[-1] != "/":
            data_path = data_path + "/"

        # for sub in test_list_withGT:
        for i in range(len(test_list_withGT)):
            try:
                sub = test_list_withGT[i]
                print("this is sub:", sub, flush=True)
                sub_path = data_path + [f for f in os.listdir(data_path) if str(sub) in f][0] + "/"
                save_sub_path = save_path + "{}_{}_{}/".format(sys.argv[1],sys.argv[2], crop_0)
                if save_sub_path[-1] != "/":
                    save_sub_path = save_sub_path + "/"
                r2,p_value = main_test_NoGT(interval_frame, img_grad, True, sub_path, save_sub_path, judge_rot_trans,
                                            model_rot, crop,str(sub),r2_with_all_data,combined_image)
                df_r_p[sub] = r2 + p_value

                # write out after each subject is done
                tmp = pd.DataFrame(df_r_p).transpose()
                print(tmp.head(5), flush=True)
                tmp.columns = ["rotx_r_2","rotY_r_2","rotZ_r_2","rotX_P","rotY_P","rotZ_P"]
                tmp = tmp.sort_index()
                print("saving to r-p file", flush=True)
                tmp.to_csv(save_sub_path + "{}_{}_{}_r2_p_value.csv".format(sys.argv[1],sys.argv[2],crop_0))
                
            except:
                print("something wrong with this sub:",sub, flush=True)
                pass

        # for sub in test_list_noGT:
        for i in range(len(test_list_noGT)):
            try:
                sub = test_list_noGT[i]
                print("this is sub:", sub)
                sub_path = data_path + [f for f in os.listdir(data_path) if str(sub) in f][0] + "/"
                save_sub_path = save_path + "{}_{}_{}/".format(sys.argv[1], sys.argv[2], crop_0)
                if save_sub_path[-1] != "/":
                    save_sub_path = save_sub_path + "/"
                r2,p_value = main_test_NoGT(interval_frame, img_grad, False, sub_path, save_sub_path, judge_rot_trans,
                                            model_rot,crop, str(sub),r2_with_all_data,combined_image)
                df_r_p[sub] = r2 + p_value
            except:
                pass
    else:
        if data_path[-1] != "/":
            data_path = data_path + "/"
        save_path = save_path + "{}_{}_{}_{}/".format(sys.argv[1], sys.argv[2],crop_0,save_path.split("/")[-1])
        main_test_NoGT(interval_frame, img_grad, False, data_path, save_path, judge_rot_trans, model_rot,
                       crop,r2_with_all_data, combined_image)

#    if judge_rot_trans == "rot":
#        df_r_p = pd.DataFrame(df_r_p).transpose()
#        print(df_r_p.head(5))
#        df_r_p.columns = ["rotx_r_2","rotY_r_2","rotZ_r_2","rotX_P","rotY_P","rotZ_P"]
#        df_r_p = df_r_p.sort_index()
#        df_r_p.to_csv(save_sub_path + "{}_{}_{}_r2_p_value.csv".format(sys.argv[1],sys.argv[2],crop_0))

