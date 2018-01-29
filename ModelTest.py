from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import cv2

def DataGen(data_x,data_y,ClassNum,SampleNum,TestNum,TrainNum,GroupNum=0):
    a=np.arange(0,ClassNum)
    a=a.repeat(TrainNum)
    a=a*SampleNum
    b=np.arange(0,TrainNum)
    b=np.tile(b,ClassNum)
    ind_train=a+b

    a=np.arange(0,ClassNum)
    a=a.repeat(TestNum)
    a=a*SampleNum
    b=np.arange(TrainNum,SampleNum)
    b=np.tile(b,ClassNum)
    ind_test=a+b

    x_train=np.transpose(data_x[:,ind_train])
    y_train=np.transpose(data_y[:,ind_train])-1
    x_test=np.transpose(data_x[:,ind_test])
    y_test=np.transpose(data_y[:,ind_test])-1

    # the data, shuffled and split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape\
            (x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape\
            (x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape\
            (x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape\
            (x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train-GroupNum*ClassNum, ClassNum)
        y_test = keras.utils.to_categorical(y_test-GroupNum*ClassNum, ClassNum)

        return(x_train, x_test,y_train,y_test)



# img=x_train[500,:,:,0]
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv2.imshow('img', img*255)

if __name__ == '__main__':

    batch_size = 100
    # input image dimensions
    img_rows, img_cols = 70, 70

    ClassNum = 200
    SampleNum = 200
    TestNum = 50
    TrainNum = SampleNum - TestNum
    GroupNum=1

    matFn='E:/ProjCodeData/DataBase/mat_data/WH70_C200_S100.mat'
    data=sio.loadmat(matFn)
    data_x=data["data_x"]
    data_y=data["data_y"]

    (x_train,x_test,y_train,y_test)=\
        DataGen(data_x,data_y,ClassNum,SampleNum,TestNum,TrainNum,GroupNum)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = load_model\
        ('E:/ProjCodeData/DeepLearningModel/CNN_WH70_C200_S100.h5')
    rec = model.predict(x_test)
    pass

    # score = model.evaluate(x_test,y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
