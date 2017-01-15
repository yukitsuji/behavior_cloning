import os
import sys
from random import choice, random
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
            image_path = image_path.lstrip()
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
    
def get_model_architecture(init='glorot_uniform', input_shape=(160, 320, 3)):
    model = Sequential()
    model.add(Conv2D(32, 5, 5, input_shape=input_shape, init=init, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3, init=init))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3, init=init))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        verbose=0,
    )
    return model

def preprocessing(rgb_image, input_shape=(160, 320)):
    #normalization
    rgb_image = rgb_image / rgb_image.max() - 0.5
    rgb_image = rgb_image[55:135, :, :]
    rgb_image = cv2.resize(rgb_image, input_shape)
    return rgb_image

def augmentation_image(row, input_shape=(160, 320)):
    steering = row['steering']
    # choose camera's place
    place = choice(["center", "left", "right"])
    image = input_rgb("./data/" + row[place]).astype(np.float32)
    
    if place == "left":
        steering += 0.25
    elif place == "right":
        steering -= 0.25
        
    # flip or not
    rd = random()
    if random() < 0.5:
        image = flip_image(image)
        steering = steering * -1
    
    # brightness
    image = augment_brightness_camera_image(image)
    
    return image, steering
    
def generate_arrays_from_datasets(datasets, batch_size=128, input_shape=(160, 320)):
    epoch_num = datasets.shape[0] // batch_size
    while True:
        for epoch in range(epoch_num): # one batch
            image_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.float32)
            steering_batch = np.zeros((batch_size, ), dtype=np.float32)
            images = datasets.loc[epoch * batch_size : (epoch + 1) * batch_size -1].reset_index(drop=True)
            for index, row in images.iterrows():
                image, steering = augmentation_image(row)
                image_batch[index], steering_batch[index] = preprocessing(image, input_shape=input_shape), steering
            yield (image_batch, steering_batch)
    
if __name__ == '__main__':
    np.random.seed(0)
    try:
        DATA_PATH = './data/driving_log.csv'
        sample_data = pd.read_csv(DATA_PATH, usecols=[0, 1, 2, 3])
        sample_data['right'] = sample_data['right'].apply(lambda x: x.lstrip())
        sample_data['left'] = sample_data['left'].apply(lambda x: x.lstrip())
    except Exception as e:
        print(e)
    else:
        print("Training Models")

        # suffle datasets
        sample_data = sample_data.sample(frac=1).reset_index(drop=True)
        # Divide datasets into Training and Validation sets
        training_rate = 0.95
        training_sets = sample_data.loc[:int(training_rate * sample_data.shape[0])]
        validation_sets = sample_data.loc[int(training_rate * sample_data.shape[0]):]

        BATCH_SIZE = 128
        EPOCH = 24
        input_shape = (64, 64)
        model = get_model_architecture(input_shape=input_shape+(3,))
        training_generator = generate_arrays_from_datasets(training_sets, input_shape=input_shape)
        validation_generator = generate_arrays_from_datasets(validation_sets, input_shape=input_shape)

        samples_per_epoch = training_sets.shape[0] // BATCH_SIZE * BATCH_SIZE
        nb_val_samples = validation_sets.shape[0] // BATCH_SIZE * BATCH_SIZE
        
        model.fit_generator(
            training_generator, samples_per_epoch=samples_per_epoch, nb_epoch=EPOCH,
            validation_data=validation_generator, nb_val_samples=nb_val_samples
        )
        
        model.save_weights('model_conv3_aug_24_64.hdf5')
        json_string = model.to_json()

        with open('model_conv3_aug_24_64.json', 'w') as f:
            f.write(json_string)
        K.clear_session()

