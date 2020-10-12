"""
This script does the preprocessing of the datasets for each training.
It organizes the training set in three seperate sets, and processes the labels according to
the taxonomy previously obtained.

In order to show robustness, we implement a few structural perturbations on the test samples,
notably the distortion presentend in the supplementary material of our paper.
"""


### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.io import imread
import scipy.io
from scipy import ndimage
# loading the three datasets from the keras library
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10


def load_data(dataset ='mnist'):
    """
    a function to load each dataset in a preliminary format
    """
    #loading dataset from keras
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #rescaling the images to be in the [0,1] range
    x_train = (1./255.)*x_train
    n = x_test.shape[0]
    x_val = (1./255.)*x_test[:int(n/2)]
    x_test = (1./255.)*x_test[int(n/2):]

    #turning labels into one-hot encoding vectors
    y_test1= np.copy(y_test)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    print(y_test.shape)
    y_val = y_test[:int(n/2)]
    y_test = y_test[int(n/2):]
    print(y_val.shape)

    # reshape the greyscale images to a 3 dimensional vector for practical use of keras
    if len(x_train.shape) == 3 :
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    return(x_train,x_val,x_test,y_train,y_val,y_test,y_test1)


def data_processing(c = 1.0,m = 0.6, f = 0.2, perturbation='warp', s = 2, t = 0.5, dataset ='mnist'):
    """
    large function with a lot of subfunctions described independantly
    input : c,m,f : the proportion of coarse, middle, and fine annotations in the training and validation datasets
    output : (x_trains,x_test,x_vals,y_trains,y_test,y_vals) : the appropriate datasets
    """


    # FIRST Preprocessing : reshaping / definition of the training and validation samples
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    x_train = (1./255.)*x_train
    n = x_test.shape[0]
    x_val = (1./255.)*x_test[:int(n/2)]
    x_test = (1./255.)*x_test[int(n/2):]

    y_test1= np.copy(y_test)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    print(y_test.shape)
    y_val = y_test[:int(n/2)]
    y_test = y_test[int(n/2):]
    print(y_val.shape)
    if len(x_train.shape) == 3 :
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)



    def preprocessing_labels(y,c,m,f):
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
        n = y.shape[0]
        y_res1 = np.zeros((n,2))
        if dataset == 'cifar10':
            y_res2= np.zeros((n,5))
        else :
            y_res2= np.zeros((n,4))
        y_res3 = np.zeros((n,10))
        perm_mnist = [3,5,8,6,0,4,7,9,2,1]
        #perm_mnist = [0,1,2,3,4,5,6,7,8,9]
        perm_svhn = [3,5,8,6,0,4,7,9,2,1]
        perm_cifar10 = [0,8,1,9,2,6,3,5,4,7]
        perm_fmnist = [0,2,6,3,4,5,7,9,1,8]
        perm = [0,1,2,3,4,5,6,7,8,9]
        if dataset == 'cifar10':
            perm = perm_cifar10
        elif dataset == 'mnist':
            perm = perm_mnist
        elif dataset == 'fashion_mnist':
            perm = perm_fmnist
        elif dataset == 'SVHN':
            perm = perm_svhn

        if dataset == 'cifar10':
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
        else :
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
        y_final3 = {"coarse":y_res1[0:int(f*n),:], "middle":y_res2[0:int(f*n),:],  "fine":y_res3[0:int(f*n),:]}
        # if f=1 there is no remaining samples
        if f <1:
            y_final2 = {"coarse":y_res1[int(f*n):int(m*n),:], "middle":y_res2[int(f*n):int(m*n),:],  "fine":y_res3[int(f*n):int(m*n),:]}
            # if m=1 there is no remaining samples
            if m<1:
                y_final1 = {"coarse":y_res1[int(m*n):int(c*n),:], "middle":y_res2[int(m*n):int(c*n),:],  "fine":y_res3[int(m*n):int(c*n),:]}
                return(y_final1,y_final2, y_final3)
            else :
                return(y_final2,y_final3)
        else :
            return(y_final3)


    def preprocessing_data(x,c,m,f):
        """
        outputs the three training datasets for the three training steps
        """
        n = int(x.shape[0])
        x_3,x_2,x_1= x[0:int(f*n),:,:,:],x[int(f*n):int(m*n) ,:,:,:],x[int(m*n):int(c*n) ,:,:,:]
        return([x_1,x_2,x_3])
    def neg(x):
        """
        turns the image into its negative
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] += 1.0-x[i,:,:,:]
        return(tmp)
    def add_noise(x,sigma):
        """
        add noise to test data sigma being the level of noise
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] += np.random.normal(loc=0.0, scale= sigma/255., size=x[i,:,:,:].shape)

        return(tmp)
    def blur(x):
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] = ndimage.gaussian_filter(tmp[i,:,:,:], sigma=1.5)
        return(tmp)
    def mean_shift(x,delta):
        """
        mean shift perturbation with parameter delta for the offset
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] += delta*np.ones(x[i,:,:,:].shape, dtype = np.float64)
            tmp[i,:,:,:] = np.minimum(tmp[i,:,:,:],np.ones(x[i,:,:,:].shape, dtype = np.float64))
        return(tmp)


    def sym_hor(x):
        """
        symmetric transform with horizontal axis of the input images of the test set
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] = np.flipud(tmp[i,:,:,:])
        return tmp
    def sym_ver(x):
        """
        symmetric transform with vertical axis of the input images of the test set
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] = np.fliplr(tmp[i,:,:,:])
        return tmp

    def warp(x,s,t):
        """
        warping distortion to a random vector field
        """
        tmp = np.copy(x)
        def generate_vector_field(shape,s,t):
            """
            function that generates the direction maps for the distortion parameter
            input :
            - shape : is the side length of the square
            - s : is the smoothness parameter
            - t : is the intensity parameter

            output :
            - u the direction map wrt x
            - v the direction map wrt y
             """
            u = np.random.normal(0.,1.,(shape,shape))
            v = np.random.normal(0.,1.,(shape,shape))
            u = gaussian_filter(u,s)
            v = gaussian_filter(v,s)
            u = (u-np.mean(u))*(t/np.std(u))
            v = (v-np.mean(v))*(t/np.std(v))
            return(u,v)
        def bilinear_interpolate(im, xx, yy):
            """
            bilinear interpolation function
            input:
            -im : the image to interpolate
            -x : the interpolation parameters wrt to x direction
            -y : the interpolation parameters wrt to y direction

            output :
            - the bilinear interpolation
            """
            x0 = np.floor(xx).astype(int)
            x1 = x0 + 1
            y0 = np.floor(yy).astype(int)
            y1 = y0 + 1

            x0 = np.clip(x0, 0, im.shape[1]-1)
            x1 = np.clip(x1, 0, im.shape[1]-1)
            y0 = np.clip(y0, 0, im.shape[0]-1)
            y1 = np.clip(y1, 0, im.shape[0]-1)

            Ia = im[y0, x0]
            Ib = im[y1, x0]
            Ic = im[y0, x1]
            Id = im[y1, x1]

            wa = (x1-xx) * (y1-yy)
            wb = (x1-xx) * (yy-y0)
            wc = (xx-x0) * (y1-yy)
            wd = (xx-x0) * (yy-y0)

            return wa*Ia + wb*Ib + wc*Ic + wd*Id
        def generate_perturbation(A):
            """
            function that generates the actual distortion perturbation
            input :
            - A : the original image
            output :
            the distorted image

            USAGE : change parameters s and t in this function
            """
            shape = A.shape[0]

            B = np.zeros(A.shape,dtype = np.float32)
            u,v = generate_vector_field(shape,s,t)
            xx, yy = np.meshgrid(np.arange(shape), np.arange(shape))  # cartesian indexing
            res = np.zeros(A.shape)

            for i in range(A.shape[2]):
                res[:,:,i] = bilinear_interpolate(A[:,:,i], u+xx, v+yy)+np.min(A)
            return(res)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] = generate_perturbation(tmp[i,:,:,:])
        return(tmp)


    def hide_top(x):
        """
        occlusion of the top of the test images
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,0:int(x.shape[1]/2),:,:] =  np.zeros((int(x.shape[1]/2),x.shape[2],x.shape[3]))

        return tmp
    def hide_bottom(x):
        """
        occlusion of the bottom of the test images
        input images : grey scale images, values between 0. and 1.
        """
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i,int(x.shape[1]/2):,:,:] =  np.zeros((int(x.shape[1]/2),x.shape[2],x.shape[3]))

        return tmp
    def random_occlusion(x):
        """
        function that generates a random occlusion of the top portion of the image.
        by generationg a border with a random angle and a random offset, sets to zero the pixels
        over this line.
        input : x, the image
        output : res1 : the occluded image

        USAGE : you can change the offset range in parameter b
        """
        shape = x.shape[1]
        tmp = np.copy(x)
        xx, yy = np.meshgrid(np.arange(shape), np.arange(shape))
        xx -= int(shape/2.)
        yy -= int(shape/2.)
        for i in range(x.shape[0]):
            a = np.random.uniform(-1.0,1.0)
            b = np.random.uniform(-2,2)
            res = a*xx +b -yy
            res = np.clip(-res, 0.,1.)

            for j in range(tmp.shape[3]):
                tmp[i,:,:,j] = res*tmp[i,:,:,j]
        return tmp

    ### APPLY PERTURBATION TO TEST SET ###
    print("Adding perturbation to test set ...")
    if perturbation == "warp":
        x_test = warp(x_test,s,t)
    elif perturbation == "hide_top":
        x_test = hide_top(x_test)
    elif perturbation == "hide_bottom":
        x_test = hide_bottom(x_test)
    elif perturbation == "random_occlusion":
        x_test = random_occlusion(x_test)
    elif perturbation == "sym_ver":
        x_test = sym_ver(x_test)
    elif perturbation == "sym_hor":
        x_test = sym_hor(x_test)
    elif perturbation == "blur":
        x_test = blur(x_test)
    ### PREPROCESS THE REMAINING DATA
    print("Preprocessing the data ...")
    x_trains = preprocessing_data(x_train,c,m,f)
    x_vals= preprocessing_data(x_val,c,m,f)
    y_trains = preprocessing_labels(y_train,c,m,f)
    y_test = preprocessing_labels(y_test,1,1,1)
    y_vals = preprocessing_labels(y_val,c,m,f)

    return(x_trains,x_vals,x_test,y_trains,y_vals,y_test)
