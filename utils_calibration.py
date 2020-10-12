import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import scipy.misc
import math
import os
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from time import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D

def preprocessing_labels1(y,c = 1.,m = 0.6, f = 0.2 ,dataset = 'mnist'):
    """
    function that organizes the labels in the appropriate way for the training and validation datasets
    input :
    - y : the original labels
    - c,m,f : the proportion of coarse, middle, fine labels in training and validation

    output :
    a tuple containing the labels for the three training steps : 1st on coarse, 2nd on coarse&middle, 3rd on all labels
    each element of this tuple is a dictionnary containing the labels for each task :
    if a task is not to be trained the labels will be an array of zeros. The loss function takes that into account
    """
    perm_mnist = [3,5,8,6,0,4,7,9,2,1]
    perm_fmnist = [0,2,6,3,4,5,7,9,1,8]
    perm = [0,1,2,3,4,5,6,7,8,9]
    perm_cifar10 = [0,8,1,9,2,6,3,5,4,7]
    n = y.shape[0]
    y_res1 = np.zeros((int(c*n),2))
    print(int(c*n))
    y_res3 = np.zeros((int(f*n),10))
    if dataset == 'cifar10':
        perm = perm_cifar10
    elif dataset == 'mnist':
        perm = perm_mnist
    elif dataset == 'fashion_mnist':
        perm = perm_fmnist
    if dataset == 'cifar10':
        y_res2= np.zeros((int(m*n),5))
        for i in range(n):
            if i< int(c*n):
                if np.argmax(y[i]) in [0,1,8,9]:
                    y_res1[i,0] = 1
                else :
                    y_res1[i,1] = 1
            if i<int(m*n):
                if np.argmax(y[i]) in [0,8]:
                    y_res2[i,0] = 1
                elif np.argmax(y[i]) in [1,9]:
                    y_res2[i,1] = 1
                elif np.argmax(y[i]) in [2,6]:
                    y_res2[i,2] = 1
                elif np.argmax(y[i]) in [3,5]:
                    y_res2[i,3] = 1
                elif np.argmax(y[i]) in [4,7]:
                    y_res2[i,4] = 1
            if i<int(f*n):
                y_res3[i,np.argmax(y[i])] = 1
        return(y_res1,y_res2,y_res3)
    else :
        y_res2= np.zeros((int(m*n),4))
        for i in range(n):
            if i< int(c*n):
                if np.argmax(y[i]) in perm[0:5]:
                    y_res1[i,0] = 1
                else :
                    y_res1[i,1] = 1
            if i<int(m*n):
                if np.argmax(y[i]) in perm[0:3]:
                    y_res2[i,0] = 1
                elif np.argmax(y[i]) in perm[3:5]:
                    y_res2[i,1] = 1
                elif np.argmax(y[i]) in perm[5:8]:
                    y_res2[i,2] = 1
                elif np.argmax(y[i]) in perm[8:]:
                    y_res2[i,3] = 1
            if i<int(f*n):
                y_res3[i,np.argmax(y[i])] = 1
        return(y_res1,y_res2,y_res3)

def normalisation_l2(x):
    """
    normalize the output of the penultimate fully connected layers with the l2 norm
    """
    res = np.zeros(x.shape)
    print(x.shape)
    for i in range(x.shape[0]):
        res[i] = x[i]/(np.linalg.norm(x[i],2)+1e-5)
    std = res.std()
    mean = res.mean()
    print("normalisation done")
    return(mean,std,res)

def normalisation_l_inf(x):
    """
    normalize the output of the penultimate fully connected layers with the l2 norm
    """
    res = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i,j] = x[i,j]/(np.max(x[i,j])+1e-5)
    return(res)
def add_uniform(x,y,n_u):
    """
    given x the normalized output of the penultimate fully connected layers,y the corresponding labels,
    this function adds n_u uniformly generated samples over the hypersphere of dimension n.
    These samples are attributed to a supplementary "unknown sample" class.

    Input :
    -the pair (x,y)
    -the number of samples to add

    Output :
    -the resulting pair (resx, resy)
    """
    n = x.shape[0]
    print(x.shape)
    print(y.shape)
    d = x.shape[1]
    new_shape = (n_u+n,d)
    resx = np.zeros(new_shape, dtype = np.float32)
    #x_norm = normalisation_l_inf(x)
    _,_,x_norm = normalisation_l2(x)
    resx[:n] = x_norm
    resx[n:] = normalisation_l2(np.random.uniform(low = 0, high = 1.0,size = (n_u,d)))[2]
    resy = np.zeros((n_u+n,y.shape[1]+1))
    resy[:n,0:y.shape[1]] = y
    resy[n:,y.shape[1]] = 1
    print("added uniform")
    print("new shape"+ str(resx.shape))
    return(resx,resy)
def shuffle(x,y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    resx = x[indices]
    resy = y[indices]
    return(resx,resy)
def add_rejection_class_test(y):
    #reshape y for practical reasons
    n,m = y.shape
    res = np.zeros((n,m+1))
    res[:,:-1] = y
    return(res)



def coarse_fc(n):
    """
    function that codes the last fully connected layer for the coarse ouptut.
    We add an output class corresponding to the rejection class presented in the calibration step
    of the paper.
    """
    input_shape = (n,)
    input_vec = Input(shape=input_shape,name = "Input1", dtype = 'float32')
    dense = Dense(3,name = "fc1",trainable=True)(input_vec)
    res = Activation('softmax',name = 'coarse')(dense)
    model_c = Model(inputs=input_vec,outputs=[res,dense])
    return(model_c)

def middle_fc(n,dataset):
    """
    function that codes the last fully connected layer for the middle ouptut.
    We add an output class corresponding to the rejection class presented in the calibration step
    of the paper.
    """
    input_shape = (n,)
    input_vec = Input(shape=input_shape,name = "Input1", dtype = 'float32')
    if dataset == 'cifar10':
        dense = Dense(6,name = "fc1",trainable=True)(input_vec)
    else :
        dense = Dense(5,name = "fc1",trainable=True)(input_vec)
    res = Activation('softmax',name = 'middle')(dense)
    model_m = Model(inputs=input_vec,outputs=[res,dense])
    return(model_m)

def fine_fc(n):
    """
    function that codes the last fully connected layer for the middle ouptut.
    We add an output class corresponding to the rejection class presented in the calibration step
    of the paper.
    """
    input_shape = (n,)
    input_vec = Input(shape=input_shape,name = "Input1", dtype = 'float32')
    #dropout = Dropout(rate = 0.2)(input_vec)
    dense = Dense(11,name = "fc1",trainable=True)(input_vec)
    res = Activation('softmax',name = 'fine')(dense)
    model_m = Model(inputs=input_vec,outputs=[res,dense])
    return(model_m)

def softmax(x):
    #numpy implementation of the softmax
    n = x.size
    res = np.exp(x)/(np.exp(x).sum())
    return(res)
def temperature_scaling(x,t):
    """
    given the output of the network before the softmax activation, this function
    rescales the output with a temperature parameter T, and then applies the softmax
    application. This scaling procedure was presented by Guo et al in the paper :
    On Calibration of Modern Neural Networks [2017]
    """
    n,d = x.shape
    res = np.copy(x)

    res *= (1./t)
    for i in range(n):
        res[i] = softmax(res[i])
    return(res)

def optimal_scale(n,pred,true):
    """
    given the output of the network before the softmax activation, this function finds the
    best scaling parameter to calibrate the network based on the ECE metric. This metric evaluates
    how the distribution of the accuracy and the distribution of confidence are close to each other.
    This is done by binning samples over there confidence, and computing the average accuracy on this bin.

    In order to find the best temperature, we evaluate the ECE for a range of 30 values of T between 1 and 3,
    and we select the one that minimizes the ECE.
    """
    def ECE(n,pred,true):
        n_bins = n
        bins = [[] for i in range(n_bins)]

        # computing the bins
        for i in range(pred.shape[0]):
            for j in range(n_bins):
                if pred[i].max()>j*(1./n_bins) and pred[i].max()<=(j+1)*(1./n_bins):
                    bins[j].append(i)
        # computing the average accuracy over the bins
        cum_sum = [0 for i in range(n_bins)]
        for j in range(n_bins):
            for i in range(len(bins[j])):
                if np.argmax(pred[bins[j][i]]) == np.argmax(true[bins[j][i]]):
                    cum_sum[j]+= 1./len(bins[j])
        # computing the ECE metric as presented in the paper
        ECE = 0.
        for j in range(n_bins):
            ECE+= abs((j+1./2)*(1./n_bins)-cum_sum[j])*(float(len(bins[j]))/2000.)
        return(ECE)

    # the range of temperature for which we evaluate the ECE
    Ts = np.linspace(1,3,30)
    l = []
    for i in range(30):
        print(Ts[i])
        scaled = temperature_scaling(pred,Ts[i])
        l.append(ECE(n,scaled,true))
    l = np.array(l)
    print(l)
    res = temperature_scaling(pred,Ts[np.argmin(l)])
    return(res,Ts[np.argmin(l)])
""" The next block of the script presents functions for combining the output of the network, some of which are not
presented in the paper, for compactness regards.  """


def weighted_majority_vote(c_pred,m_pred,f_pred,acc_c,acc_m,acc_f, dataset):
    """
    combination method that computes the WMV. Basically a MV, but weighted according to the accuracy of the classifiers.
    This is an adaptation of the same method presented by Kuncheva in "Combining pattern classifiers : methods an algorithms"[2004],
    to our problem of nested learning.
    """
    c,m,f = np.argmax(c_pred),np.argmax(m_pred),np.argmax(f_pred)
    coarse = np.zeros(2)
    middle = np.zeros(4)
    fine = np.zeros(10)

    if dataset == 'cifar10':
        middle = np.zeros(5)
    coarse[c] = 1
    middle[m] = 1
    fine[f] = 1
    res = np.zeros(10)
    w1 = np.log(acc_c/(1.-acc_c))
    w2 = np.log(acc_m/(1.-acc_m))
    w3 = np.log(acc_f/(1.-acc_f))
    if dataset == 'cifar10':
        for i in range(10):
            if i <2:
                res[i] = w1*coarse[0] + w2*middle[0] + w3*fine[i]
            elif 2<=i <4:
                res[i] = w1*coarse[0] + w2*middle[1] + w3*fine[i]
            elif 4 <=i<6:
                res[i] = w1*coarse[1] + w2*middle[2] + w3*fine[i]
            elif 6<=i<8:
                res[i] = w1*coarse[1] + w2*middle[3] + w3*fine[i]
            else:
                res[i] = w1*coarse[1] + w2*middle[4] + w3*fine[i]
    else :
        for i in range(10):
            if i <3:
                res[i] = w1*coarse[0] + w2*middle[0] + w3*fine[i]
            elif 3<=i <5:
                res[i] = w1*coarse[0] + w2*middle[1] + w3*fine[i]
            elif 5 <=i<8:
                res[i] = w1*coarse[1] + w2*middle[2] + w3*fine[i]
            else:
                res[i] = w1*coarse[1] + w2*middle[3] + w3*fine[i]
    index = np.argmax(res)
    return(index)
def majority_vote(c_pred,m_pred,f_pred,dataset):
    """
    combination method that computes the Majprity Vote.
    This is an adaptation of the same method presented by Kuncheva in "Combining pattern classifiers : methods an algorithms"[2004],
    to our problem of nested learning.
    """
    c,m,f = np.argmax(c_pred),np.argmax(m_pred),np.argmax(f_pred)
    coarse = np.zeros(2)
    middle = np.zeros(4)
    fine = np.zeros(10)
    if dataset == 'cifar10':
        middle = np.zeros(5)
    coarse[c] = 1
    middle[m] = 1
    fine[f] = 1
    res = np.zeros(10)
    if dataset == 'cifar10':
        for i in range(10):
            if i <2:
                res[i] = coarse[0] + middle[0] + fine[i]
            elif 2<=i <4:
                res[i] = coarse[0] + middle[1] + fine[i]
            elif 4 <=i<6:
                res[i] = coarse[1] + middle[2] + fine[i]
            elif 6<=i<8:
                res[i] = coarse[1] + middle[3] + fine[i]
            else:
                res[i] = coarse[1] + middle[4] + fine[i]
    else :
        for i in range(10):
            if i <3:
                res[i] = coarse[0] + middle[0] + fine[i]
            elif 3<=i <5:
                res[i] = coarse[0] + middle[1] + fine[i]
            elif 5 <=i<8:
                res[i] = coarse[1] + middle[2] + fine[i]
            else:
                res[i] = coarse[1] + middle[3] + fine[i]
    index = np.argmax(res)
    return(index)

def acc_one_hot(true_f,pred):
    #accuracy of one-hot predictors
    acc = 0
    for i in range(2000):
        if np.argmax(true_f[i]) == pred[i]:
            acc +=1./2000
    return(acc)

def acc_function(true_f, pred):
    #accuracy of probabilistic outputs
    acc = 0
    for i in range(2000):
        if np.argmax(true_f[i]) == np.argmax(pred[i]):
            acc +=1./2000
    return(acc)

def proba(c_pred,m_pred,f_pred, dataset):
    """
    This function encodes the combination method that we developped for the fine output.
    Details in the paper on the theoretical justification of this method.
    """
    p = np.zeros(10)
    if dataset == 'cifar10':
        for i in range(10):
            if i <4:
                if i <2:
                    p[i] = c_pred[0]*(m_pred[0]/(m_pred[0]+m_pred[1]))*(f_pred[i]/np.sum(f_pred[0:2]))
                elif i <4:
                    p[i] = c_pred[0]*(m_pred[1]/(m_pred[0]+m_pred[1]))*(f_pred[i]/np.sum(f_pred[2:4]))
            if i >=4:
                if i <6:
                    p[i] = c_pred[1]*(m_pred[2]/(m_pred[2]+m_pred[3]+m_pred[4]))*(f_pred[i]/np.sum(f_pred[4:6]))
                elif i <8:
                    p[i] = c_pred[1]*(m_pred[3]/(m_pred[2]+m_pred[3]+m_pred[4]))*(f_pred[i]/np.sum(f_pred[6:8]))
                elif i <10:
                    p[i] = c_pred[1]*(m_pred[4]/(m_pred[2]+m_pred[3]+m_pred[4]))*(f_pred[i]/np.sum(f_pred[8:10]))
    else :
        for i in range(10):
            if i <5:
                if i <3:
                    p[i] = c_pred[0]*(m_pred[0]/(m_pred[0]+m_pred[1]))*(f_pred[i]/np.sum(f_pred[0:3]))
                elif i <5:
                    p[i] = c_pred[0]*(m_pred[1]/(m_pred[0]+m_pred[1]))*(f_pred[i]/np.sum(f_pred[3:5]))
            if i >=5:
                if i <8:
                    p[i] = c_pred[1]*(m_pred[2]/(m_pred[2]+m_pred[3]))*(f_pred[i]/np.sum(f_pred[5:8]))
                elif i <10:
                    p[i] = c_pred[1]*(m_pred[3]/(m_pred[2]+m_pred[3]))*(f_pred[i]/np.sum(f_pred[8:]))
    return(p)
def proba_middle(c_pred,m_pred,dataset):
    """
    This function encodes the combination method that we developped for the middle output.
    Details in the paper on the theoretical justification of this method.
    """
    if dataset =='cifar10':
            p = np.zeros(5)
            for i in range(5):
                if i in [0,1]:
                    p[i] = c_pred[0]*(m_pred[i]/(m_pred[0]+m_pred[1]))
                if i in [2,3,4]:
                    p[i] = c_pred[1]*(m_pred[i]/(m_pred[2]+m_pred[3]++m_pred[4]))
    else :
        p = np.zeros(4)
        for i in range(4):
            if i in [0,1]:
                p[i] = c_pred[0]*(m_pred[i]/(m_pred[0]+m_pred[1]))
            if i in [2,3]:
                p[i] = c_pred[1]*(m_pred[i]/(m_pred[2]+m_pred[3]))
    return(p)



def proba_fc(c_pred,f_pred,dataset):
    """
    This function encodes the combination method that we developped for the fine output, but only combines fine and coarse outputs.
    Details in the paper on the theoretical justification of this method.
    """
    p = np.zeros(10)
    for i in range(10):
        if dataset =='cifar10':
            if i <4:
                p[i] = (c_pred[0])*(f_pred[i]/np.sum(f_pred[0:4]))
            else:
                p[i] = (c_pred[1])*(f_pred[i]/np.sum(f_pred[4:]))
        else:
            if i<5:
                p[i] = (c_pred[0])*(f_pred[i]/np.sum(f_pred[0:5]))
            else:
                p[i] = (c_pred[1])*(f_pred[i]/np.sum(f_pred[5:]))
    return(p)
def proba_fm(m_pred,f_pred, dataset):
    """
    This function encodes the combination method that we developped for the fine output, but only combines fine and middle outputs.
    Details in the paper on the theoretical justification of this method.
    """
    p = np.zeros(10)
    if dataset == 'cifar10':
        for i in range(10):
            if i <4:
                if i <2:
                    p[i] = (m_pred[0])*(f_pred[i]/np.sum(f_pred[0:2]))
                else:
                    p[i] = (m_pred[1])*(f_pred[i]/np.sum(f_pred[2:4]))
            else:
                if i <6:
                    p[i] = (m_pred[2])*(f_pred[i]/np.sum(f_pred[4:6]))
                elif i <8:
                    p[i] = (m_pred[3])*(f_pred[i]/np.sum(f_pred[6:8]))
                else:
                    p[i] = (m_pred[4])*(f_pred[i]/np.sum(f_pred[8:]))
    else :
        for i in range(10):
            if i <5:
                if i <3:
                    p[i] = (m_pred[0])*(f_pred[i]/np.sum(f_pred[0:3]))
                else:
                    p[i] = (m_pred[1])*(f_pred[i]/np.sum(f_pred[3:5]))
            else:
                if i <8:
                    p[i] = (m_pred[2])*(f_pred[i]/np.sum(f_pred[5:8]))
                else:
                    p[i] = (m_pred[3])*(f_pred[i]/np.sum(f_pred[8:]))
    return(p)

def finetocoarse(f_pred,dataset):
    # this function computes the coarse prediction inferred from the fine predictions
    res = np.zeros(2)
    for i in range(10):
        if dataset =='cifar10':
            if i <4:
                res[0]+=f_pred[i]
            else :
                res[1]+=f_pred[i]
        else:
            if i <5:
                res[0]+=f_pred[i]
            else :
                res[1]+=f_pred[i]
    return(res)

def middletocoarse(m_pred):
    # this function computes the coarse prediction inferred from the middle predictions
    res = np.zeros(2)
    for i in range(m_pred.size):
        if i in [0,1]:
            res[0]+=m_pred[i]
        else :
            res[1]+=m_pred[i]
    return(res)

def finetomiddle(f_pred,dataset):
    # this function computes the middle prediction inferred from the fine predictions
    res = np.zeros(4)

    if dataset == 'cifar10':
        res = np.zeros(5)
        for i in range(10):
            if i <2:
                res[0]+=f_pred[i]
            elif i <4:
                res[1]+=f_pred[i]
            elif i <6:
                res[2]+=f_pred[i]
            elif i <8:
                res[3]+=f_pred[i]
            elif i >=8:
                res[4]+=f_pred[i]
    else :
        for i in range(10):
            if i <3:
                res[0]+=f_pred[i]
            elif i <5:
                res[1]+=f_pred[i]
            elif i <8:
                res[2]+=f_pred[i]
            elif i >=8:
                res[3]+=f_pred[i]
    return(res)

def acc(y_pred,y_true):
    res = 0
    for i in range(2000):
        if np.argmax(y_true[i]) == np.argmax(y_pred[i]):
            res +=1./2000
    return(res)
