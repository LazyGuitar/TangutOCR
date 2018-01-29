

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

def DataGen(data_x,data_y,ClassNum,SampleNum,TestNum,TrainNum, GroupInd=0):
    StartInd = int(ClassNum * SampleNum * GroupInd)

    a=np.arange(0,ClassNum)
    a=a.repeat(TrainNum)
    a=a*SampleNum
    b=np.arange(0,TrainNum)
    b=np.tile(b, ClassNum)
    ind_train = a+b
    # ind_train += StartInd

    a=np.arange(0,ClassNum)
    a=a.repeat(TestNum)
    a=a*SampleNum
    b=np.arange(TrainNum,SampleNum)
    b=np.tile(b,ClassNum)
    ind_test = a+b
    # ind_test += StartInd

    x_train=np.transpose(data_x[:,ind_train])
    y_train=np.transpose(data_y[:,ind_train])-1
    x_test=np.transpose(data_x[:,ind_test])
    y_test=np.transpose(data_y[:,ind_test])-1
    y_test -= ClassNum*GroupInd
    y_train -= ClassNum * GroupInd

    # the data, shuffled and split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1,
                                  img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1,
                                img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0],
                                  img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0],
                                img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, ClassNum)
        y_test = keras.utils.to_categorical(y_test, ClassNum)

        return(x_train, x_test,y_train,y_test,input_shape)

if __name__ == '__main__':
    ClassNum=50
    SampleNum=200
    TestNum=50
    TrainNum=SampleNum-TestNum
    GroupInd=3

    batch_size = 50
    epochs = 12
    # input image dimensions
    img_rows, img_cols = 70, 70

    matFn="E:/ProjCodeData/DataBase/mat_data/WH70_C668_S200_%d.mat"%(GroupInd)
    modFn="E:/ProjCodeData/DeepLearningModel/CNN_WH70_C668_S200_%d.h5"%(GroupInd)


    # matFn='E:/ProjCodeData/DataBase/mat_data/WH70_C668_S200_1.mat'
    data=sio.loadmat(matFn)
    data_x=data["data_x"]
    data_y=data["data_y"]
    (x_train,x_test,y_train,y_test,input_shape)=\
    DataGen(data_x, data_y,ClassNum, SampleNum,
            TestNum, TrainNum, GroupInd)
    # '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(ClassNum, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    model.save(modFn)
    # '''
    # model = load_model\
    # ('E:/ProjCodeData/DeepLearningModel/CNN_WH70_C200_S100_3.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
