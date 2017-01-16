# behaivior_cloning
A car trained by Convolutional Network to drive itself in a video game

## Movie

## Dependencies
python3 / Keras / Tensorflow / numpy / pandas / opencv3

## Datasets
- size: 8036
- original image shape: (160, 320, 3)

## Data Augmentation
#### Horizontal Flip
#### Changing Brightness 

## Preprocessing
#### Normalization
#### Crop and Resize images
- crop image from (160, 320, 3) to (80, 320, 3)

- Resize image for model architecture
from (80, 320, 3) to (64, 64, 3)  

## Model Architecture
simple convolutional network [Image of Model Architecture](./model.png") 
- Max Pooling
- Batch Normalization
- ReLU
- Adam


<img src="./model.png" width="200px;"/>

### Hyperparameter for training model
- Batch Size: 128
- EPOCH : 24
- learning rate: 0.001
