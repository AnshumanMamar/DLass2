# CS6910 Assignment 2

Convolution Neural Network

# PART-A 

## CNN Training Script

This script enables the training of a Convolutional Neural Network (CNN) using PyTorch, providing flexibility in configuring the network architecture, training settings, and hyperparameters through command-line arguments. 


## CNN Architecture

The architecture of the Convolutional Neural Network  is outlined within the script named PartA.ipynb. This architecture is composed of several convolutional layers, each followed by max-pooling layers to reduce dimensionality. Finally, there is a densely connected layer. The structure of this CNN can be tailored to 
specific requirements by adjusting various parameters, offering flexibility in its design to accommodate different datasets and objectives.

## Usage

1. To begin, ensure you have your dataset prepared and accessible. Update the data_path argument in the script to point to the directory containing your dataset.
2. Once your dataset is ready and the data_path argument is set, execute the training script with the desired configurations. 

## command line arguments that can be passed to set parameters
-   Kernel Size : '-ks', '--kernel_size', [3,5] etc
-   Activation: '-a', '--activation', ['ReLU', 'GELU' , 'LeakyReLU' , "SiLU" , "Mish" "elu"]
-   Epochs: '-e', '--epochs', [25,20] etc
-   Dense layer size: '-neu','--dense_nodes', [1024,512] etc
-   kernel Organisation : '-ko', '--kernel_organisation' ,  ['same','double','half','default']
-   Num Of Kernels : '-nk' , '--num_kernel' , [16,32] etc
**Default set to my best configuration**

## Prediction Function

The function model_predict is designed to accept an image as input and utilize the trained Convolutional Neural Network (CNN) model to make a prediction regarding the image's class label. 

## Training Process
In the training process we opitimized the CNN prameters and minimized the cross entropy loss across predicted and actual labels. It invlolved of interating on the 
dataset for many epochs,then updating model parameters with the help of backpropagation, and evaluating the model's performance.

## Plotting Grid

The function plot_grid is responsible for generating a visual representation of the model's predictions on a subset of the validation dataset. It creates a grid layout with 10 rows and 3 columns, where each cell contains an image along with its corresponding true label and the label predicted by the model. 

# PART B

## Function for  Data Loading (`data_load`):

This segment of code is responsible for loading the dataset from the designated directory and applying various transformations to the data to prepare it for training. These transformations typically include resizing the images to a uniform size, performing normalization to ensure consistent data scales.

## Function for Training (`Train`):

This function serves the purpose of training the model over a specified number of epochs. It involves iterating through the dataset, during which the model conducts a forward pass to generate predictions for each batch of data. Subsequently, it computes the loss between the predicted labels and the ground truth labels. 



## Function for Layer Freezing  (`freeze`):

This function provides the capability to freeze specific layers within the model based on a specified strategy. The freezing process entails preventing the parameters within certain layers from being updated during the training phase. Users can choose to freeze either the first k layers, middle layers, or the last k layers of the model, depending on their preferences and requirements. 



