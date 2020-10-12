
"""
calibration& combination file :
Following the model of main script, this script takes various arguments and can be either used
at traintime or test time.
It first performs the 2 calibration steps :
- mitigating with the uniform samples
- temperature scaling
To train those parameters, on should set the parameter --traintime True.

Then, the second blocks performs the combination of the calibrated outputs as presented in the paper.
This script heavily depends on functions coded in the utils_calibration.py file
 """
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import scipy.misc
import math
import os
import pandas as pd
from tensorflow.keras.datasets import mnist


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7*1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


## importing useful functions
from utils_calibration import *
from preprocessing import *


## defining the arguments for the calibration step
parser = argparse.ArgumentParser(description='Calibration and combination')
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='the dataset on which to execute the program')
parser.add_argument('--traintime', type=bool, default=False, metavar='Tr',
                    help='True at train time, False at test time ')
parser.add_argument('--model-id', type=int, default=0, metavar='MID',
                    help='the id of the trained model')
parser.add_argument('--coarse-rate', type=float, default=100, metavar='C',
                    help='100 times the proportion of coarse labels')
parser.add_argument('--middle-rate', type=float, default=60, metavar='M',
                    help='100 times the proportion of fine labels')
parser.add_argument('--fine-rate', type=float, default=20, metavar='F',
                    help='100 times the proportion of fine labels')
parser.add_argument('--perturbation', type=str, default='hide_top', metavar='P',
                    help='the perturbation to add to the test set')
parser.add_argument('--s', type=float, default=2, metavar='S',
                    help='the smoothing parameter of the distortion')
parser.add_argument('--t', type=float, default=0.5, metavar='T',
                    help='the intensity parameter of the distortion')
parser.add_argument('--single', type=bool, default=False, metavar='SI',
                    help='True if training the single model network')
parser.add_argument('--FT', type=bool, default=False, metavar='FT',
                    help='True if fine tuning the fine classification')

args = parser.parse_args()
id = args.model_id
c = args.coarse_rate
m = args.middle_rate
f = args.fine_rate
s = args.s
t = args.t

#laoding data
(x_train,x_val,x_test,y_train,y_val,y_test,y_test1) = load_data(dataset =args.dataset)
_,_,x_tests,_,_,y_tests = data_processing(c/100.,m/100.,f/100.,args.perturbation,args.s,args.t, dataset = args.dataset)
print(x_test.shape)
print(x_tests.shape)

#setting perturbation name
per_name = ''
if args.perturbation=='warp':
    per_name = "warp_s{}_t{}".format(args.s,args.t)
else :
    per_name = args.perturbation

#defining the good model
if args.dataset == 'mnist' :
    from models_mnist_adv import *
    fine_model = Mnist_classifier_fine_bn()
elif args.dataset == 'fashion_mnist' :
    from models_fashion_mnist import *
    fine_model = FashionMnist_classifier_fine_bn()
elif args.dataset == 'cifar10' :
    from models_cifar10 import *
    fine_model = fine_modelUnet_AP()


### PREPROCESSING THE DATASET
n = x_train.shape[0]
c1 = c/100.
m1 = m/100.
f1 = f/100.
x_coarse = x_train[:int(c1*n)]
x_middle = x_train[:int(m1*n)]
x_fine = x_train[:int(f1*n)]

y_coarse,y_middle, y_fine = preprocessing_labels1(y_train,c = c/100.,m = m/100., f = f/100., dataset = args.dataset )


### Compiling the chosen model
optimizer = tf.keras.optimizers.Adam(lr = 2.5e-3)
fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle":'categorical_crossentropy',"fine": 'categorical_crossentropy' },
loss_weights = {"coarse":0.01*10,"middle":0.01*10,"fine":3.3},optimizer= optimizer ,metrics=['accuracy'])
if args.FT == True :
    fine_model.load_weights('weights/weights_{}/model_multi_outputfine_tuned{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c,m,f),by_name = True)
else :
    fine_model.load_weights('weights/weights_{}/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c,m,f),by_name = True)

def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)  
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = fine_model(image)[2]
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)   
    signed_grad = tf.sign(gradient)    
    return signed_grad
eps = 0.025
perturbation = adversarial_pattern(x_tests, y_tests['fine']).numpy()
x_tests += eps*perturbation


# predicting the output for the test set
fine_model.evaluate(x_tests,y_tests)
print('Predictions ongoing...')
X_feat1 = fine_model.predict(x_coarse)
X_feat2 = fine_model.predict(x_middle)
X_feat3 = fine_model.predict(x_fine)
X_feat_test = fine_model.predict(x_tests)
X_feat_test_original = fine_model.predict(x_test)
print('Predictions Done.')

# extracting the penultimate feature before the last fully connected layer for each granularity level for the train set
feat_coarse = X_feat1[3]
feat_middle = X_feat2[4]
feat_fine = X_feat3[5]
# extracting the penultimate feature before the last fully connected layer for each granularity level for the original test set
test_coarse_0 = X_feat_test_original[3]
test_middle_0 = X_feat_test_original[4]
test_fine_0 = X_feat_test_original[5]
# extracting the penultimate feature before the last fully connected layer for each granularity level for the perturbed test set
test_coarse = X_feat_test[3]
test_middle = X_feat_test[4]
test_fine = X_feat_test[5]

# reshaping the labels with the rejection class
yc,ym,yf = preprocessing_labels1(y_test,c = 1.,m = 1., f = 1., dataset = args.dataset )
y_test_fine = add_rejection_class_test(yf)
y_test_middle = add_rejection_class_test(ym)
y_test_coarse = add_rejection_class_test(yc)

# declaring the model to be retrained for the last fully connected layer
size_c = test_coarse_0.shape[1]
size_m = test_middle_0.shape[1]
size_f = test_fine_0.shape[1]
model_c = coarse_fc(size_c)
model_m = middle_fc(size_m,args.dataset)
model_f = fine_fc(size_f)

def n_uniform(n,s):
    return(int(n*(1.2**(s/2))))

#compilation
model_c.compile(loss = {'coarse' : 'categorical_crossentropy'},optimizer= optimizer, metrics = ['accuracy'] )
model_m.compile(loss = {'middle':'categorical_crossentropy'},optimizer= optimizer, metrics = ['accuracy'] )
model_f.compile(loss = {'fine':'categorical_crossentropy'},optimizer= optimizer, metrics = ['accuracy'] )

# adding the rejection class training samples
# resx_c,resy_c = add_uniform(feat_coarse,y_coarse, n_uniform(feat_coarse.shape[0],size_c))
# resx_m,resy_m = add_uniform(feat_middle,y_middle, n_uniform(feat_middle.shape[0],size_m))
# resx_f,resy_f = add_uniform(feat_fine,y_fine, n_uniform(feat_fine.shape[0],size_f))
resx_c,resy_c = add_uniform(feat_coarse,y_coarse, 50000)
resx_m,resy_m = add_uniform(feat_middle,y_middle, 60000)
resx_f,resy_f = add_uniform(feat_fine,y_fine, 80000)

print(resx_c.shape)
#training if at traintime
if args.traintime :
    trainx,trainy = shuffle(resx_c[1000:],resy_c[1000:])
    valx,valy = resx_c[:1000],resy_c[:1000]
    model_c.fit(trainx, trainy,
              batch_size=32,
              epochs=15,
              verbose=1,
              validation_data = (valx,valy),
             shuffle = True)
    model_c.save_weights('weights/weights_{}/coarse_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))
    trainx,trainy = shuffle(resx_m[1000:],resy_m[1000:])
    valx,valy = resx_m[:1000],resy_m[:1000]
    model_m.fit(trainx,trainy,
              batch_size=32,
              epochs=12,
              verbose=1,
              validation_data = (valx,valy),
             shuffle = True)
    model_m.save_weights('weights/weights_{}/middle_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))
    trainx,trainy = shuffle(resx_f[1000:],resy_f[1000:])
    valx,valy = resx_f[:1000],resy_f[:1000]
    model_f.fit(trainx,trainy,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_data = (valx,valy),
             shuffle = True)
    model_f.save_weights('weights/weights_{}/fine_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))

#loading the trained weights
model_c.load_weights('weights/weights_{}/coarse_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))
model_m.load_weights('weights/weights_{}/middle_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))
model_f.load_weights('weights/weights_{}/fine_last_layer_{}_id{}c{}m{}f{}.h5'.format(args.dataset,args.dataset,id,c,m,f))

#predicting the ouptut of the coarse fully connected layer before and after the activation + evaluating the new fully connected layer
#for the clean data
means,stds,test_coarse10 = normalisation_l2(test_coarse_0)
[resc_00,resc_0] = model_c.predict(test_coarse10)
mean_conf_c_or = np.max(resc_00[:-1],axis = 1).mean()
mean_acc_c_or = model_c.evaluate(test_coarse10,y_test_coarse)[-1]

#predicting the ouptut of the middle fully connected layer before and after the activation + evaluating the new fully connected layer
#for the clean data
means,stds,test_middle10 = normalisation_l2(test_middle_0)
[resm_00,resm_0] = model_m.predict(test_middle10)
mean_conf_m_or = np.max(resm_00[:-1],axis = 1).mean()
mean_acc_m_or = model_m.evaluate(test_middle10,y_test_middle)[-1]

#predicting the ouptut of the fine fully connected layer before and after the activation + evaluating the new fully connected layer
#for the clean data
means,stds,test_fine10 = normalisation_l2(test_fine_0)
[resf_00,resf_0] = model_f.predict(test_fine10)
mean_conf_f_or = np.max(resf_00[:-1],axis = 1).mean()
mean_acc_f_or = model_f.evaluate(test_fine10,y_test_fine)[-1]

print(test_coarse.shape)
print(y_test_coarse.shape)
#predicting the ouptut of the coarse fully connected layer before and after the activation + evaluating the new fully connected layer
#for the perturbed data
means,stds,test_coarse1 = normalisation_l2(test_coarse)
[resc_01,resc_0_t] = model_c.predict(test_coarse1)
mean_conf_c_per = np.max(resc_01[:-1],axis = 1).mean()
mean_acc_c_per = model_c.evaluate(test_coarse1,y_test_coarse)[-1]

#predicting the ouptut of the middle fully connected layer before and after the activation + evaluating the new fully connected layer
#for the perturbed data
means,stds,test_middle1 = normalisation_l2(test_middle)
[resm_01,resm_0_t] = model_m.predict(test_middle1)
mean_conf_m_per = np.max(resm_01[:-1],axis = 1).mean()
mean_acc_m_per = model_m.evaluate(test_middle1,y_test_middle)[-1]

#predicting the ouptut of the fine fully connected layer before and after the activation + evaluating the new fully connected layer
#for the perturbed data
means,stds,test_fine1 = normalisation_l2(test_fine)
[resf_01,resf_0_t] = model_f.predict(test_fine1)
mean_conf_f_per = np.max(resf_01[:-1],axis = 1).mean()
mean_acc_f_per = model_f.evaluate(test_fine1,y_test_fine)[-1]


#creating a result file for the calibration and combination step
if not os.path.exists('results/calibrated_results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f)):
    df = pd.DataFrame(columns=['perturbation','coarse_acc','middle_acc','fine_acc','MV','WMV','Mean','Product','C&F','M&F','C&M&F','Middle'])
    df.to_csv('results/calibrated_results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), index=False)
df = pd.read_csv('results/calibrated_results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), engine='python')


print(mean_acc_c_per)
print(mean_acc_m_per)
print(mean_acc_f_per)
print(mean_conf_c_per)
print(mean_conf_m_per)
print(mean_conf_f_per)

print(mean_acc_c_or)
print(mean_acc_m_or)
print(mean_acc_f_or)
print(mean_conf_c_or)
print(mean_conf_m_or)
print(mean_conf_f_or)

if args.traintime :
    #finding the best temperature parameters on the original distribution
    resc_scaled_original ,T_coarse = optimal_scale(40,resc_0,y_test_coarse)
    resm_scaled_original ,T_middle = optimal_scale(40,resm_0,y_test_middle)
    resf_scaled_original, T_fine = optimal_scale(40,resf_0,y_test_fine)
    T_s = np.array([T_coarse,T_middle, T_fine])
    np.save('weights/weights_{}/temperatures_{}_id{}c{}m{}f{}.npy'.format(args.dataset,args.dataset,id,c,m,f), T_s)

#loading temperature parameters
T_s = np.load('weights/weights_{}/temperatures_{}_id{}c{}m{}f{}.npy'.format(args.dataset,args.dataset,id,c,m,f))
T_coarse = T_s[0]
T_middle = T_s[1]
T_fine = T_s[2]
print(T_coarse)
print(T_middle)
print(T_fine)

#temperature scaling
resc_scaled_original = temperature_scaling(resc_0,T_coarse)
resm_scaled_original = temperature_scaling(resm_0,T_middle)
resf_scaled_original = temperature_scaling(resf_0,T_fine)

resc_scaled = temperature_scaling(resc_0_t,T_coarse)
resm_scaled = temperature_scaling(resm_0_t,T_middle)
resf_scaled = temperature_scaling(resf_0_t,T_fine)

""" The following block is for combination """

# setting the appropriate permutation list
perm = [0,1,2,3,4,5,6,7,8,9]
perm_mnist = [3,5,8,6,0,4,7,9,2,1]
perm_svhn = [3,5,8,6,0,4,7,9,2,1]
perm_cifar10 = [0,8,1,9,2,6,3,5,4,7]
perm_fmnist = [0,2,6,3,4,5,7,9,1,8]

if args.dataset == 'cifar10':
    perm = perm_cifar10
elif args.dataset == 'mnist':
    perm = perm_mnist
elif args.dataset == 'fashion_mnist':
    perm = perm_fmnist
elif args.dataset == 'SVHN':
    perm = perm_svhn

# reshaping and renormalising the outputs of the last layer so that dimensions match in order to combine them
if args.dataset =='cifar10':
    resc_2fine = np.array([np.array([0.25*resc_scaled[i,0] for k in range(4)]+[0.166*resc_scaled[i,1] for k in range(6)]) for i in range(2000)])
    resm_2fine = 0.5*np.array([np.array([resm_scaled[i,0] for k in range(2)]
                               +[resm_scaled[i,1] for k in range(2)]+
                               [resm_scaled[i,2] for k in range(2)]+
                               [resm_scaled[i,3]for k in range(2)]+
                               [resm_scaled[i,4]for k in range(2)]) for i in range(2000)])
    resc_2fine_original = np.array([np.array([0.25*resc_scaled_original[i,0] for k in range(4)]+[0.166*resc_scaled_original[i,1] for k in range(6)]) for i in range(2000)])
    resm_2fine_original = 0.5*np.array([np.array([resm_scaled_original[i,0] for k in range(2)]+[resm_scaled_original[i,1] for k in range(2)]+[resm_scaled_original[i,2] for k in range(2)]+[resm_scaled_original[i,3] for k in range(2)]+[resm_scaled_original[i,4] for k in range(2)]) for i in range(2000)])

else :
    resc_2fine = 0.2*np.array([np.array([resc_scaled[i,0] for k in range(5)]+[resc_scaled[i,1] for k in range(5)]) for i in range(2000)])
    resm_2fine = 0.5*np.array([np.array([resm_scaled[i,0] for k in range(3)]+[resm_scaled[i,1] for k in range(2)]+[resm_scaled[i,2] for k in range(3)]+[resm_scaled[i,3]for k in range(2)]) for i in range(2000)])
    resc_2fine_original = 0.2*np.array([np.array([resc_scaled_original[i,0] for k in range(5)]+[resc_scaled_original[i,1] for k in range(5)]) for i in range(2000)])
    resm_2fine_original = 0.5*np.array([np.array([resm_scaled_original[i,0] for k in range(3)]+[resm_scaled_original[i,1] for k in range(2)]+[resm_scaled_original[i,2] for k in range(3)]+[resm_scaled_original[i,3] for k in range(2)]) for i in range(2000)])

print(perm)
resf_perm = np.array([np.array([resf_scaled[i,perm[k]] for k in range(10)]) for i in range(2000)])
resf_perm_original = np.array([np.array([resf_scaled_original[i,perm[k]] for k in range(10)]) for i in range(2000)])
true_f = np.array([np.array([y_test_fine[i,perm[k]] for k in range(10)]) for i in range(2000)])
true_c = np.array([np.array([y_test_coarse[i,k] for k in range(2)]) for i in range(2000)])
true_m = np.array([np.array([y_test_middle[i,k] for k in range(4)]) for i in range(2000)])
print("accuracy perturbed")
print(acc(resc_scaled[:,:-1],true_c))
print(acc(resm_scaled[:,:-1],true_m))
print(acc(resf_perm,true_f))
print("accuracy normal")
print(acc(resc_scaled_original[:,:-1],true_c))
print(acc(resm_scaled_original[:,:-1],true_m))
print(acc(resf_perm_original,true_f))
## one-hot encoding combination methods
print("one hot encoding techniques")
pred = [weighted_majority_vote(resc_scaled[i,:-1],resm_scaled[i,:-1],resf_perm[i,:],mean_acc_c_per,mean_acc_m_per,mean_acc_f_per,args.dataset) for i in range(2000)]
acc_wmv = (acc_one_hot(true_f,pred))
print(acc_wmv)
pred = [majority_vote(resc_scaled[i,:-1],resm_scaled[i,:-1],resf_perm[i,:], args.dataset) for i in range(2000)]
acc_mv = (acc_one_hot(true_f,pred))
print(acc_mv)
pred = [weighted_majority_vote(resc_scaled_original[i,:-1],resm_scaled_original[i,:-1],resf_perm_original[i,:],mean_acc_c_or,mean_acc_m_or,mean_acc_f_or,args.dataset) for i in range(2000)]
print(acc_one_hot(true_f,pred))
pred = [majority_vote(resc_scaled_original[i,:-1],resm_scaled_original[i,:-1],resf_perm_original[i,:],args.dataset) for i in range(2000)]


## probability vectors combination methods
ps = [proba(resc_scaled[i,:-1],resm_scaled[i,:-1],resf_perm[i,:],args.dataset) for i in range(2000)]
pms = [proba_middle(resc_scaled[i,:-1],resm_scaled[i,:-1],args.dataset) for i in range(2000)]
fcs = [finetocoarse(resf_perm[i,:],args.dataset) for i in range(2000)]
mcs = [middletocoarse(resm_scaled[i,:-1]) for i in range(2000)]
fms = [finetomiddle(resf_perm[i,:],args.dataset) for i in range(2000)]


l = [0 for i in range(2000)]
acc_cmf = 0
for i in range(len(l)):
    if np.argmax(true_f[i]) == np.argmax(ps[i]):
        acc_cmf +=1./2000.
print("proba combination")
print(acc_cmf)

acc_mc = 0
for i in range(len(l)):
    if np.argmax(y_test_middle[i]) == np.argmax(pms[i]):
        acc_mc +=1./len(l)
print("proba combination middle")
print(acc_mc)

acc_f2m = 0
for i in range(len(l)):
    if np.argmax(y_test_middle[i]) == np.argmax(fms[i]):
        acc_f2m +=1./len(l)
print("fine 2 middle")
print(acc_f2m)

acc_f2c = 0
for i in range(len(l)):
    if np.argmax(y_test_coarse[i]) == np.argmax(fcs[i]):
        acc_f2c +=1./len(l)
print("fine 2 coarse")
print(acc_f2c)

acc_m2c = 0
for i in range(len(l)):
    if np.argmax(y_test_coarse[i]) == np.argmax(mcs[i]):
        acc_m2c +=1./len(l)
print("middle 2 coarse")
print(acc_m2c)

ps = [proba_fc(resc_scaled[i,:-1],resf_perm[i,:],args.dataset) for i in range(2000)]
acc_fc = 0
for i in range(len(l)):
    if np.argmax(true_f[i]) == np.argmax(ps[i]):
        acc_fc +=1./len(l)
print("proba fine and coarse")
print(acc_fc)

ps = [proba_fm(resm_scaled[i,:-1],resf_perm[i,:],args.dataset) for i in range(2000)]
acc_fm = 0
for i in range(len(l)):
    if np.argmax(true_f[i]) == np.argmax(ps[i]):
        acc_fm +=1./len(l)
print("proba fine and middle")
print(acc_fm)

ps = [proba(resc_scaled[i,:-1],resm_scaled[i,:-1],resf_perm[i,:],args.dataset) for i in range(2000)]
acc = 0
for i in range(len(l)):
    if np.argmax(true_f[i]) == np.argmax(ps[i]):
        acc +=1./len(l)
print("proba combination")
print(acc)


mean_pred = (1./3)*(resc_2fine + resm_2fine + resf_perm)
acc_mean = 0
for i in range(2000):
    if np.argmax(true_f[i]) == np.argmax(mean_pred[i]):
        acc_mean +=1./2000
print("mean")
print(acc_mean)


print("prod")
prod_pred = np.zeros((2000,10))
acc_prod = 0
for i in range(2000):
    for j in range(10):
        prod_pred[i,j] = resc_2fine[i,j]*resm_2fine[i,j]*resf_perm[i,j]
    if np.argmax(true_f[i]) == np.argmax(prod_pred[i]):
        acc_prod +=1./len(l)
print(acc_prod)

## storing all these results in a new file
df = df.append({'perturbation':per_name, 'coarse_acc':mean_acc_c_per,'middle_acc':mean_acc_m_per,'fine_acc':mean_acc_f_per, 'MV':acc_mv,'WMV':acc_wmv,'Mean':acc_mean,'Product':acc_prod,'C&F':acc_fc,'M&F':acc_fm,'C&M&F':acc_cmf , 'Middle' : acc_mc }, ignore_index=True)
df.to_csv('results/calibrated_results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), index=False)
