# Keras and Tenserflow implementation for the paper : Nested Learning for multi-granular tasks
This github repository corresponds to the code for our ICASSP 2021 submission.
The goal of this open source code is to show how we implemented the concept of nested learning for simple classification problems, but also to encourage the reader to try its own architecture. 
The architecture we chose is indeed not the main contribution of the paper, so you are welcome to try any new architecture that somehow stick to the general framework presented in the third section of the paper 
(Nested Learning), described by the following image.
![Architecture](https://github.com/nestedlearning2019/code_iclr/blob/master/framework.png) "Illustrative scheme of the proposed framework"    

From left to right, the input data x, a first set of layers that extract from X a feature representation f<sub> 1 </sub>;
  which leads to 	&#374;<sub> 1 </sub>; (estimation of the coarse label Y<sub> 1 </sub>). f<sub> 1 </sub> is then jointly exploited in addition with complementary information of the input. 
  This leads to a second representation f<sub> 2 </sub> from which a finer classification is obtained. The same idea is repeated until the fine level of classification is achieved.

We will present in this readme the general conditions to run the code, for MNIST, CIFAR10, and Fashion-MNIST.

## Preparing the environment
In order to run the code, you should read this section and make sure your python environment fullfills all the following requirements. Using **GPUs** and **Cuda 7.0** is highly recommended for computation time purposes. 
* Download this directory 
* Make sure **Keras** and **Tensorflow** are installed in your python environment
* In this directory, create the following directory, using these terminal instructions : 
```
#directory that stores the result csv files
$mkdir results
#directory that stores the Tensorboard runs
$mkdir log_dir
#directory that stores the weights
$mkdir weights
## the following lines depend on which dataset you want to test the model 
$cd weights
$mkdir weights_mnist
$mkdir weights_fashion_mnist
$mkdir weights_cifar10
```


## How to run the code
The 2 important files to run to get results are `main2.py` and `calibration.py`. The order in which you run those two is crucial.
### Running `main2.py`
This script can either train or test the model chosen for the chosen dataset. The models are defined in the files `models_[dataset].py`, and you can write your own models in those files. You should first run the model in train mode, because the test mode relies on already trained model which do not exist if the model isn't trained.

#### Training
Let's think about an example. We want to try the model we designed for the **MNIST dataset**, with **30 %** of only coarsely annotated samples from the original dataset (2 categories, 18000 samples), **30 %** of coarse and intermediate data (samples annotated both with 2 categories and 4 categories, 18000 samples), and **20%** of coarse middle and fine data (samples annotated  with 2 coarse categories, 4 internediate categories and 10 fine classes, 12000 samples). In order to train this model, copy paste this in the terminal :
```
$python3 main2.py --dataset "mnist" --traintime True --model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 
```
The **model-id** parameter is convenient so that you know which trained model you are testing afterwards if you want to compute several trainings. In order to change the training hyper-parameters, you can also add the corresponding arguments with the desired values, which are also presented in the parser.
Running this line will have several consequences:
* creating 3 **".h5"** file for the trained model, one for each step of the training. The information about the training of this model are stored in its name. Each of those files store the resulting trained weights.
* creating a log_dir sub directory with the training details for Tensorboard.
#### Testing
Now that the model is trained, we can test it. For that we can test several perturbations, all coded in the `preprocessing.py` file. Let's say we want to try both the results of the model on the original distribution, and also on the images distorted with parameter (S,T) = (1.0,1.0). You can run the following lines :
```
$python3 main2.py --dataset "mnist"--model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "original"
$python3 main2.py --dataset "mnist"--model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "warp" --s 1.0 --t 1.0
```
Running this two lines will result in the creation of a csv file in the results directory with the name **'results\_mnist\_id0\_c80\.0\_m50.0\_f20.0.csv'**. Each line of this .csv file stores the accuracies and confidences of the coarse, middle and fine classifier, the perturbation type we tested the model on, and the type of model (single output vs single output).

### Running `calibration.py`

Once the training of the model is done with the `main2.py` execution, we can try the calibration and combination methods. To calibrate the results, we also need a train and test step. It is because we need to fit the calibration parameters on the original distribution, and not on the test distribution which could have been disturbed by one of the coded functions. However, the train mode also tests our combination methods on the original distribution. This parameter is here so that we don't need to retrain everything at every run.

#### Training

Following the same example as before, we now want to calibrate the model. To do that we can run the following command line :
```
$python3 calibration.py --dataset "mnist" --traintime True --model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "original" 
```
This will result in the following updates :
* the weights **"weights/weights\_mnist/\[granularity level\]\_last\_layer\_id0\_c80.0\_m50.0\_20.0.h5"** will be created and trained.
* The temperature parameter of the scaling step will be created and optimized, and stored in the corresponding weights files.
* The calibrated + combination result file for this model will be created and updated with the results of the combination methods. These results are stored in **"results/calibrated\_results\_mnist\_id0\_c80.0\_m50.0\_20.0.csv"**. Each line of this file contains the results of the calibration with accuracies and confidences before combinations, the type of perturbation tested, and the accuracies for all the combination methods coded in our `utils_calibration.py` file.

#### Testing

Now that the calibration parameters are fitted, we can test the method on perturbed data. By running this line for example :
```
$python3 calibration.py --dataset "mnist" --model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "warp" --s 1.0 --t 1.0
```
we add the corresponding line to **"results/calibrated\_results\_mnist\_id0\_c80.0\_m50.0\_20.0.csv"**.
