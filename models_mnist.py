import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate, Lambda, Softmax, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import backend as K
from time import time

def Mnist_classifier_coarse_bn():
    """
    The architecture of the coarse model. We name each layer so that it is easier to reload wih keras
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)

    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)

    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)

    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
    conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=False)(conv6)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=False))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)


    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=False)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)
    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
    conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=False)(conv8)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=False))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=False)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)



def Mnist_classifier_middle_bn():
    """
    The architecture of the coarse & middle model.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)

    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)

    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)

    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
    conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=True)(conv6)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=True))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)

    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=True)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)
    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
    conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=False)(conv8)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=False))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=False)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)


def Mnist_classifier_fine_bn():
    """
    The architecture of the coarse & middle & fine model.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)
    conv4 = Dropout(0.25)(conv4)
    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)
    conv5 = Dropout(0.25)(conv5)
    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)

    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
    conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=True)(conv6)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=True))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)

    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=True)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)
    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
    conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=True)(conv8)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=True))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=True)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)

def Mnist_classifier_full_bn():
    """
    The architecture of the single-output model.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)
    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)
    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)
    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)
    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)
    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
    conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=True)(conv6)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=True))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)
    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
    conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=True)(conv8)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=True))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)
    res3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=True)(res3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = res3
    model = Model(inputs=input_img,outputs=final_result)
    return(model)



def bottleneck_coarse_bn():
    """
    The architecture of the coarse model without Skipped-connections.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)
    conv4 = Dropout(0.25)(conv4)
    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)
    conv5 = Dropout(0.25)(conv5)
    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)
    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=False)(conv5bis)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=False))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)

    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=False)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)


    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=False)(conv7bis)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=False))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=False)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)



def bottleneck_middle_bn():
    """
    The architecture of the coarse & middle model without skipped-connections.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)
    conv4 = Dropout(0.25)(conv4)
    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)
    conv5 = Dropout(0.25)(conv5)
    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)
    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=True)(conv5bis)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=True))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)

    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=True)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)


    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=False)(conv7bis)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=False))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=False)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)


def bottleneck_fine_bn():
    """
    The architecture of the coarse & middle & fine model without skipped-connections.
    """
    input_shape = (28,28,1)
    input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
    conv1 = Conv2D(2, (3, 3),padding='same',name = "conv2d_1", trainable=True)(input_img)
    conv1 = (BatchNormalization(name= 'batch_normalization' ))(conv1)
    conv1 = (Activation('relu',name = 'activation'))(conv1)

    conv2 = Conv2D(4, (3, 3),padding='same',name = "conv2d_2", trainable=True)(conv1)
    conv2 = (BatchNormalization(name= 'batch_normalization_1' ))(conv2)
    conv2 = (Activation('relu',name = 'activation_1'))(conv2)

    conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
    conv3 = Conv2D(8, (3, 3),padding='same',name = "conv2d_3", trainable=True)(conv2bis)
    conv3 = (BatchNormalization(name= 'batch_normalization_2' ))(conv3)
    conv3 = (Activation('relu',name = 'activation_2'))(conv3)

    conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
    conv4 = Conv2D(16, (3, 3),padding='same',name = "conv2d_4", trainable=True)(conv3bis)
    conv4 = (BatchNormalization(name= 'batch_normalization_3' ))(conv4)
    conv4 = (Activation('relu',name = 'activation_3'))(conv4)
    conv4 = Dropout(0.25)(conv4)
    conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
    conv5 = Conv2D(32, (3, 3),padding='same',name = "conv2d_5", trainable=True)(conv4bis)
    conv5 = (BatchNormalization(name= 'batch_normalization_4' ))(conv5)
    conv5 = (Activation('relu',name = 'activation_4'))(conv5)
    conv5 = Dropout(0.25)(conv5)
    d1 = GlobalAveragePooling2D()(conv5)
    res1 = Dense(2,name = "fc1")(d1)
    res1 = Activation('softmax',name = 'coarse')(res1)
    conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
    conv7 = Conv2D(16, (3, 3),padding='same',name = "conv2d_6",trainable=True)(conv5bis)
    conv7 = (BatchNormalization(name= 'batch_normalization_5' ,trainable=True))(conv7)
    conv7 = (Activation('relu',name = 'activation_5'))(conv7)

    d2 = GlobalAveragePooling2D()(conv7)
    res2 = Dense(4,name = "fc2",trainable=True)(d2)
    res2 = Activation('softmax',name = 'middle')(res2)


    conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
    conv9 = Conv2D(16, (3, 3),padding='same',name = "conv2d_7",trainable=True)(conv7bis)
    conv9 = (BatchNormalization(name= 'batch_normalization_6' ,trainable=True))(conv9)
    conv9 = (Activation('relu',name = 'activation_6'))(conv9)

    d3 = GlobalAveragePooling2D()(conv9)
    res3 = Dense(10,name = "fc3", trainable=True)(d3)
    res3 = Activation('softmax',name = 'fine')(res3)
    final_result = [res1,res2,res3,d1,d2,d3]
    model = Model(inputs=input_img,outputs=final_result)
    return(model)
