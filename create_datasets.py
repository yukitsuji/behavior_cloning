import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import img_to_array, load_img

def input_rgb(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    return image

def flip_image(image):
    return cv2.flip(image, 1)

def augment_brightness_camera_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def save_image_from_numpy(image, path):
    image = Image.fromarray(np.uint8(image))
    image.save(path)

def augmentation_datasets(images_path, steering_sets, add_steering=0.25, image_shape=(160, 320, 3), place='center', folder_path='./data/', new_data_path="./data/new_driving_log.csv"):
    with open(new_data_path, 'a') as f:
        steering_sets += add_steering
        images_path = images_path.apply(lambda x: x.lstrip())

        df = pd.DataFrame({
            'image' : images_path,
            'steering' : steering_sets,  
        })
        if place == 'center':
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)
            
        for name in ["FLIP_", "BRIGHT_", "FLIP_BRIGHT_"]:
            path = name + images_path
            df["image"] = path
            if name == "BRIGHT_":
                df["steering"] = steering_sets
            else:
                df["steering"] = steering_sets * -1
            df.to_csv(f, header=False)
            
            dir_path = folder_path + name + "IMG"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        
        for index, image_path in enumerate(images_path):
            image = input_rgb(folder_path + image_path).astype(np.uint8)
            fliped_image = flip_image(image)
            path = folder_path + "FLIP_" + image_path
            save_image_from_numpy(fliped_image, path)

            bright_image = augment_brightness_camera_image(image)
            path = folder_path + "BRIGHT_" + image_path
            save_image_from_numpy(bright_image, path)
            
            fliped_bright_image = augment_brightness_camera_image(fliped_image)
            path = folder_path + "FLIP_BRIGHT_" + image_path
            save_image_from_numpy(fliped_bright_image, path)   
        
def create_datasets(sample_data):
    NEW_DATA_PATH = "./data/new_driving_log.csv"
    
    if os.path.exists(NEW_DATA_PATH):
        os.remove(NEW_DATA_PATH)
        
    steering_sets = sample_data['steering']

    augmentation_datasets(sample_data['center'], steering_sets, add_steering=0, new_data_path=NEW_DATA_PATH)
    augmentation_datasets(sample_data['left'], steering_sets, add_steering=0.25, new_data_path=NEW_DATA_PATH, place='left')
    augmentation_datasets(sample_data['right'], steering_sets, add_steering=-0.25, new_data_path=NEW_DATA_PATH, place='right')
        
np.random.seed(0)
K.clear_session()
DATA_PATH = './data/driving_log.csv'
sample_data = pd.read_csv(DATA_PATH, usecols=[0, 1, 2, 3])
create_datasets(sample_data)