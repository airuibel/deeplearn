# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:12:27 2018

@author: sh02060
"""

import numpy as np  
import dill  
filename= 'globalsave.pkl'  
#dill.dump_session(filename) 
##载入数据集  
dill.load_session(filename)  

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(j, y, test_size=0.33, random_state=42)
Y_train = np_utils.to_categorical(Y_train,2)
Y_test = np_utils.to_categorical(Y_test, 2)

# 全局变量
batch_size = 4
nb_classes = 2
epochs = 15
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

input_shape = ( img_rows, img_cols,1)

#构建模型
model = Sequential()

"""
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
"""
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape)) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
model.add(Activation('relu')) #激活层
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(50)) #全连接层1
model.add(Activation('relu')) #激活层
model.add(Dropout(0.25)) #随机失活
model.add(Dense(2)) #全连接层2
model.add(Activation('softmax')) #Softmax评分
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
#评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
##网络结构
model.summary()
