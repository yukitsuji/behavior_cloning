# behaivior_cloning


## Movie
[![ScreenShot](http://img.youtube.com/vi/qPCW-x0oUvI/0.jpg)](https://youtu.be/qPCW-x0oUvI)



## Dependencies
python3 / Keras / Tensorflow / numpy / pandas / opencv3

## Datasets
- size: 8036
- original image shape: (160, 320, 3)

## Data Augmentation
Original Datasets was made at Track 1, which environment is stable in terms of Brightness, Shadow, and so on.
And to make left turn and right the same number, Flip images randomly.

#### Horizontal Flip
#### Changing Brightness 

## Preprocessing
#### Normalization
#### Crop and Resize images
- crop image from (160, 320, 3) to (80, 320, 3)

- Resize image for model architecture
from (80, 320, 3) to (64, 64, 3)  

## Excecution


## Model Architecture
simple convolutional network [Image of Model Architecture](https://github.com/yukitsuji/behaivior_cloning/blob/master/model.png) 
- Max Pooling
- Batch Normalization
- ReLU
- Adam

### Hyperparameter for training model
- Batch Size: 128
- EPOCH : 24
- learning rate: 0.001
