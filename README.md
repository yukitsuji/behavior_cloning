# About this project
Project 3 of Udacity's self driving car project, behavior cloning for driving.  
The main task is to drive a car around in a simulator on a race track,  
and then use deep learning and related techniques to mimic the behavior of human.  
Reference: [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  

## Movie
[![ScreenShot](http://img.youtube.com/vi/qPCW-x0oUvI/0.jpg)](https://youtu.be/qPCW-x0oUvI)

## Dependencies
python3 / Keras / Tensorflow / numpy / pandas / opencv3

# Model Architecture
simple convolutional network: [click](https://github.com/yukitsuji/behaivior_cloning/blob/master/model.png) image  
For avoid overfitting, I use *Batch Normalization*.  
The way of Network initialization is *Glorot Initialization*.  
When I changed final architecture like adding one Fully Connected Layer,  
The model was be overfitting to the training cource.  
Conversely, when I changed 1st Conv Layer's kernel (3, 3),  
The model was underfitting to the training cource.  

- Input: [Batch size, 64, 64, 3]
- 1st layer: Convolution Layer  
  - Conv Layer
    - kernel size: (5, 5)
    - kenel num: 32
    - border_mode: same
  - Activate Function: ReLU
  - Batch Normalization
  - Max Pooling Layer
    - kernel size: (2, 2)
- 2st layer: Convolution Layer
  - Conv Layer
    - kernel size: (3, 3)
    - kenel num: 32
    - border_mode: same
  - Activate Function: ReLU
  - Batch Normalization
  - Max Pooling Layer
    - kernel size: (2, 2)
- 3st layer: Convolution Layer
  - Conv Layer
    - kernel size: (3, 3)
    - kenel num: 32
    - border_mode: same
  - Activate Function: ReLU
  - Batch Normalization
  - Max Pooling Layer
    - kernel size: (2, 2)
- Flatten
- 4th layer: Fully Connected Layer
  - output dimension: 128
  - Activate Function: ReLU
  - Batch Normalization
- Output Layer: Fully Connected Layer
  - output dimension: 1
  - loss: Mean Squared Error

### Hyperparameter for training model
For tuning the model, I experiment different learning rate, epoch.  
About learning rate, I select 0.001 from [0.001, 0.0005, 0.0001], because other values are so small, and learning speed was so slow/ But I think learning rate was affected by the way of network initialization.

About epoch, validation error about 12 epoch was so good too.  
But when apply it to test cource, the car run windingly.  
From 24 to 32 epochs, behavior cloning car could complete test course.  
- optimizer: *Adam*
- learning rate: *0.001*
- Batch Size: *128*
- EPOCH : *24*

## Datasets
- size: 8036(each camera has)  
- original image shape: (160, 320, 3)

From Datasets, I useed "center/left/right Image" and "steering"  

# Data Augmentation
Original Datasets was made at Track 1, which environment is stable in terms of Brightness, Shadow, and so on.  
To avoid overfitting original datasets, I augment datasets by two techniques.  

## choose camera place
Datasets have image from "center", "left", "right" place.  
By using "left" and "right" camera, The model could learn the situation how to recover from side.  
So when choosing left camera, add steering value to 0.25.  
When choosing right camera, subtract steering value to 0.25.  

## Horizontal Flip
to make left turn and right the same number, Flip images randomly.  

## Changing Brightness
Original Datasets was made at Track 1, which environment is stable in terms of Brightness, Shadow, and so on.  
So, I convert RGB image to HSV, and change value of V dimension.  
After that, I convert HSV to RGB images.    
By execute this, avoid overfitting of light condision.  

# Preprocessing
## Normalization
- rescale image from [0-255] to [-0.5-0.5] by divide by max value of image and subtract 0.5

## Crop and Resize images
- crop image from (160, 320, 3) to (80, 320, 3)
- Resize image for model architecture
from (80, 320, 3) to (64, 64, 3)  

## How to Excecute
I utilize Keras's fit_generator function because there are many data.  
By using fit_generator, I can augment on the fly, preprocess images and train the model batch-by-batch  
