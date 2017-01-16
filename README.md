# About this project
Project 3 of Udacity's self driving car project, behavior cloning for driving.  
The main task is to drive a car around in a simulator on a race track, and then use deep learning to mimic the behavior of human.

## Movie
[![ScreenShot](http://img.youtube.com/vi/qPCW-x0oUvI/0.jpg)](https://youtu.be/qPCW-x0oUvI)

## Dependencies
python3 / Keras / Tensorflow / numpy / pandas / opencv3

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
By fit_generator, I can augment on the fly, preprocess images and train the model batch-by-batch  

# Model Architecture
simple convolutional network: [click](https://github.com/yukitsuji/behaivior_cloning/blob/master/model.png) image
- Max Pooling
- Batch Normalization
- ReLU
- Adam

### Hyperparameter for training model
- Batch Size: 128
- EPOCH : 24
- learning rate: 0.001
