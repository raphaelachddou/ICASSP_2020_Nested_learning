
"""
main file :
This script can be used for both training and testing of our method of nested learning.
It inherits the model(...).py files, the preprocessing.py file, and some useful functions defined in utils.py
To run this code, I suggest to run it in a terminal with the appropriate parameters.
This script can be tested on three datasets : cifar10, Fashion-mnist, and Mnist.

This script can perform the training and the testing of our method BUT NOT the combination and calibration step,
which is in the calibration.py file.
In order to test a model you need to first execute the file to train the model and then
test with the option --traintime True.

As explained in the paper, we compare two types of training : an end to end training with a single fine grained output,
and a cascaded training with three outputs of different granularity.
To enable the single output mode, use the parameter --single True .

Also, we compare the models with or without skipped-connection for the MNIST dataset.
This is handled by the --bottleneck parameter which should be set to True if we want
to use the model without skipped connections.
 """

### IMPORTS
import argparse
import scipy.io
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from time import time
from preprocessing import *
from utils import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3*1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

### SETTING PARAMETERS
parser = argparse.ArgumentParser(description='Experiments for CIFAR10')
## general arguments
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='the dataset on which to execute the program')
parser.add_argument('--traintime', type=bool, default=False, metavar='T',
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
parser.add_argument('--t', type=float, default=0.5, metavar='I',
                    help='the intensity parameter of the distortion')
## arguments for the training process
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--coarse-epochs', type=int, default=100, metavar='CE',
                    help='the number of epochs')
parser.add_argument('----middle-epochs', type=int, default=100, metavar='ME',
                    help='the number of epochs')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='the number of epochs')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',
                    help='input batch learning rate for training (default: 1e-3)')
parser.add_argument('--opt', type=str, default='Adam', metavar='O',
                    help='the chosen optimizer for training')
parser.add_argument('--FT', type=bool, default=False, metavar='FT',
                    help='True if fine tuning the fine classification')
parser.add_argument('--single', type=bool, default=False, metavar='SI',
                    help='True if training the single model network')
parser.add_argument('--bottleneck', type=bool, default=False, metavar='BO',
                    help='True if the model is trained as a bottleneck')

args = parser.parse_args()
id = args.model_id
c = args.coarse_rate
m = args.middle_rate
f = args.fine_rate
s = args.s
t = args.t
## defining the models according to the chosen dataset and the chosen architecture : single vs 3 output
print(args.single)
if args.dataset == 'mnist' :
    from models_mnist import *
    coarse_model = Mnist_classifier_coarse_bn()
    middle_model = Mnist_classifier_middle_bn()
    fine_model = Mnist_classifier_fine_bn()
    if args.bottleneck:
        coarse_model = bottleneck_coarse_bn()
        middle_model = bottleneck_middle_bn()
        fine_model = bottleneck_fine_bn()
    single_model = Mnist_classifier_full_bn()
elif args.dataset == 'fashion_mnist' :
    from models_fashion_mnist_seq import *
    coarse_model = FashionMnist_classifier_coarse_bn()
    middle_model = FashionMnist_classifier_middle_bn()
    fine_model = FashionMnist_classifier_fine_bn()
    single_model = FashionMnist_classifier_full_bn()

if args.dataset == 'cifar10' :
    from models_cifar10 import *
# coarse_model = Cifar10_classifier_coarse_bn()
# middle_model = Cifar10_classifier_middle_bn()
# fine_model = Cifar10_classifier_fine_bn()
# single_model = Cifar10_classifier_full_bn()    
    coarse_model = coarse_modelUnet_AP()
    middle_model = middle_modelUnet_AP()
    fine_model = fine_modelUnet_AP()
    single_model = fine_modelUnet_single_AP()

### defining the datasets using a function defined in preprocessing.py
x_trains,x_vals,x_test,y_trains,y_vals,y_test = data_processing(c/100, m/100, f/100,args.perturbation, s, t, args.dataset)



if args.traintime == True :
    """
    The following block is for training the models
    """
    ### creating the logdir to use Tensorboard
    logdir1 = 'log_dir/logcoarse{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir2 = 'log_dir/logmiddle{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir3 = 'log_dir/logfine{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir4 = 'log_dir/logfine{}_tuning_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir5 = 'log_dir/logfine{}_single_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    if not os.path.exists(logdir1):
        os.makedirs(logdir1)
    if not os.path.exists(logdir2):
        os.makedirs(logdir2)
    if not os.path.exists(logdir3):
        os.makedirs(logdir3)
    if not os.path.exists(logdir4):
        os.makedirs(logdir4)
    if not os.path.exists(logdir5):
        os.makedirs(logdir5)

    ### defining the optimizer according to the parameters set by the user
    if args.opt =='Adam':
        optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    if args.opt =='SGD':
        optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)
    if not(args.single):
        """
        this block is for training the 3-output model (ours)
        """
        if not(args.FT):
            ### TRAINING THE COARSE BLOCK FIRST
            #defining the optimizers
            if args.opt =='Adam':
                optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
            if args.opt =='SGD':
                optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)
            #defining callbacks
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir1), EarlyStopping(monitor='val_loss', patience = 10,restore_best_weights = True)]
            #compiling the coarse block
            coarse_model.compile(loss = {"coarse": 'categorical_crossentropy'},
                loss_weights = {"coarse":1.},optimizer= optimizer ,metrics=['accuracy'])
            #training the coarse block
            coarse_model.fit(x_trains[0], y_trains[0],
                      batch_size=args.batch_size,
                      epochs=args.coarse_epochs,
                      verbose=1,
                      callbacks = callbacks,
                      validation_data=(x_vals[0], y_vals[0]))
            #saving the weights in the appropriate directory
            #each dataset has a dedicated weight directory
            coarse_model.save_weights('weights/weights_{}/model_multi_outputcoarse{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f))

            ### TRAINING THE COARSE AND MIDDLE BLOCK
            #defining the optimizers
            if args.opt =='Adam':
                optimizer = tf.keras.optimizers.Adam(lr=0.5*args.learning_rate)
            if args.opt =='SGD':
                optimizer = tf.keras.optimizers.SGD(lr=0.5*args.learning_rate)
            #defining callbacks
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir2),EarlyStopping(monitor='val_middle_loss', patience = 10,restore_best_weights = True)]
            #compiling the coarse and middle block including some multi-task loss weights
            middle_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy'},
               loss_weights = {"coarse":1.,"middle":float(c)/m},optimizer= optimizer ,metrics=['accuracy'])
            #loading the weights of the coarse training for the appropriate weights
            middle_model.load_weights('weights/weights_{}/model_multi_outputcoarse{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f),by_name = True)
            #training the coarse and middle block
            middle_model.fit(x_trains[1], y_trains[1],
                     batch_size=args.batch_size,
                     epochs=args.middle_epochs,
                     verbose=1,
                     callbacks = callbacks,
                     validation_data=(x_vals[1], y_vals[1]))
            middle_model.save_weights('weights/weights_{}/model_multi_outputmiddle{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f))
            #saving the weights in the appropriate directory

            ### TRAINING THE COARSE, MIDDLE, FINE BLOCK ALTOGETHER
            #defining the optimizers
            if args.opt =='Adam':
                optimizer = tf.keras.optimizers.Adam(lr=0.25*args.learning_rate)
            if args.opt =='SGD':
                optimizer = tf.keras.optimizers.SGD(lr=0.25*args.learning_rate)
            #defining callbacks
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir3),
            EarlyStopping(monitor='val_fine_loss',patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)]
            #compiling the coarse and middle and fine block including some multi-task loss weights
            fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine":
                                         'categorical_crossentropy'},
                loss_weights = {"coarse":1.,"middle":float(c)/m,"fine":float(c)/f},optimizer= optimizer ,metrics=['accuracy'])
            #loading the weights of the coarse and middle training for the appropriate weights
            fine_model.load_weights('weights/weights_{}/model_multi_outputmiddle{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f),by_name = True)
            #training the coarse, middle and fine block
            fine_model.fit(x_trains[2], y_trains[2],
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      verbose=1,
                      callbacks = callbacks,
                      validation_data=(x_vals[2], y_vals[2]))
            #saving the weights in the appropriate directory
            fine_model.save_weights('weights/weights_{}/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f))

        else :
            """
            This block is to fine-tune the training with more training steps,
            we didn't use it in the paper, but if optimized this could lead to an improvement
            """
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =10),
            TensorBoard(log_dir = logdir4)]
            fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine":
                                         'categorical_crossentropy'},
                loss_weights = {"coarse":1.,"middle":1.5*float(c)/m,"fine" : 2*float(c)/f},optimizer = tf.keras.optimizers.Adam(lr=0.5*args.learning_rate) ,metrics=['accuracy'])
            fine_model.summary()
            #fine_model.load_weights('weights/weights_{}/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f),by_name = True)
            fine_model.fit(x_trains[2], y_trains[2],
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      verbose=2,
                      callbacks = callbacks,
                      validation_data=(x_vals[2], y_vals[2]))
            fine_model.save_weights('weights/weights_{}/model_multi_outputfine_tuned{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f))

    else:
        """
        This block is to train the single-output network
        """
        #defining callbacks
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
        TensorBoard(log_dir = logdir5),EarlyStopping(monitor='val_loss', patience = 10,restore_best_weights = True)]
        #compiling the single output model
        single_model.compile(loss = {"fine":'categorical_crossentropy'},optimizer= tf.keras.optimizers.Adam(lr=0.25*args.learning_rate)
                           ,metrics=['accuracy'], loss_weights = {"fine":1.})
        #training the model
        single_model.fit(x_trains[2], y_trains[2]['fine'],
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=1,
                  callbacks = callbacks,
                  validation_data=(x_vals[2], y_vals[2]['fine']))
        single_model.save_weights('weights/weights_{}/model_single{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c, m,f))
        #saving the weights in the appropriate directory
else :
    """
    This block is for testing the model you want
    """

    #creating the result file with its corresponding name for the model, dataset, and ratio tested
    if not os.path.exists('results/results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f)):
        # the files should show the accuracies and confidences for each level of granularity for each tested perturbation
        df = pd.DataFrame(columns=['type','perturbation','coarse_acc','middle_acc','fine_acc','coarse_conf','middle_conf', 'fine_conf'])
        df.to_csv('results/results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), index=False)
    df = pd.read_csv('results/results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), engine='python')

    #defining the perturbation name
    per_name = ''
    if args.perturbation=='warp':
        per_name = "warp_s{}_t{}".format(args.s,args.t)
    else :
        per_name = args.perturbation

    #loading ground truth labels
    y_test_coarse = y_test['coarse']
    y_test_middle = y_test['middle']
    y_test_fine = y_test['fine']
    if not(args.single) :
        """
        TESTING the Multi-output Model
        """

        #compilation
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
        fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine": 'categorical_crossentropy' },
        loss_weights = {"coarse":1.,"middle":c/m,"fine":c/f},optimizer= optimizer ,metrics=['accuracy'])
        if args.FT == True:
            fine_model.load_weights('weights/weights_{}/model_multi_outputfine_tuned{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c,m,f), by_name = True)
        else :
            fine_model.load_weights('weights/weights_{}/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c,m,f),by_name = True)
        #prediction of the outputs for the test set
        y_pred = fine_model.predict(x_test)
        #split the predictions for each level of precision
        y_coarse = y_pred[0][:,:]
        y_middle = y_pred[1][:,:]
        y_fine = y_pred[2][:,:]

        #reshaping the groundtruth for practical use
        n = y_test_fine.shape[0]
        y_testfine1 = [np.argmax(y_test_fine[i,:]) for i in range(n)]
        y_test_coarse1 = [np.argmax(y_test_coarse[i,:]) for i in range(n)]
        y_test_middle1 = [np.argmax(y_test_middle[i,:]) for i in range(n)]

        #initialising the result variables for confidence
        fine_conf = 0.
        middle_conf = 0.
        coarse_conf = 0.
        #computing mean confidences over the test set
        for i in range(x_test.shape[0]):
            fine_conf+=np.max(y_fine[i,:])
            middle_conf+=np.max(y_middle[i,:])
            coarse_conf+=np.max(y_coarse[i,:])
        fine_conf  *= (1./x_test.shape[0])
        middle_conf  *= (1./x_test.shape[0])
        coarse_conf  *= (1./x_test.shape[0])
        print(fine_conf)
        print(middle_conf)
        print(coarse_conf)

        #computing the accuracies thanks to the acc_cmf function defined in utils.py
        acc_c,acc_m, acc_f = acc_cmf(y_coarse, y_middle, y_fine,y_testfine1 , args.dataset)
        print(acc_c)
        print(acc_m)
        print(acc_f)

        #updating the result file
        df = df.append({'type':'multi-output','perturbation':per_name, 'coarse_acc':acc_c,'middle_acc':acc_m,'fine_acc':acc_f, 'coarse_conf':coarse_conf ,'middle_conf': middle_conf,'fine_conf':fine_conf }, ignore_index=True)
        df.to_csv('results/results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), index=False)
    else:
        """
        This block is for testing the single output model
        """
        #compiling the single output model and loading the trained weights
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
        single_model.compile(loss = {"fine":'categorical_crossentropy' },loss_weights = {"fine":3.3},optimizer= optimizer ,metrics=['accuracy'])
        single_model.load_weights('weights/weights_{}/model_single{}_id{}c{}m{}f{}.h5'.format(args.dataset, args.dataset,id,c,m,f),by_name = True)
        #computing the predictions
        y_pred = single_model.predict(x_test)
        n = y_test_fine.shape[0]
        print(n)
        print(y_pred.shape)
        #computing the confidences and accuracies
        acc_c,acc_m,acc_f = acc_cmf_single(y_pred,y_test_fine,args.dataset)
        (conf_c,conf_m, conf_f) = conf_cmf_single(y_pred,args.dataset)
        print(acc_c)
        print(acc_m)
        print(acc_f)
        print(conf_c)
        print(conf_m)
        print(conf_f)
        #updating the appropriate result file : the type column indicates that the result is the one of the single-output or the multi-output
        df = df.append({'type':'single_output','perturbation':per_name, 'coarse_acc':acc_c,'middle_acc':acc_m,'fine_acc':acc_f, 'coarse_conf':conf_c ,'middle_conf': conf_m,'fine_conf':conf_f}, ignore_index=True)
        df.to_csv('results/results_{}_id{}_c{}_m{}_f{}.csv'.format(args.dataset,id,c, m,f), index=False)
