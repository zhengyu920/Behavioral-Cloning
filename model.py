
# Download the data set
from urllib.request import urlretrieve
from os.path import isfile
from os.path import exists
import zipfile
# if the zip file is not downloaded
# otherwise do nothing
if not isfile('data.zip'):
    print("start download data set")
    urlretrieve('https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip','data.zip')
    print('data downloaded.')
# unzip the data
if not exists('data'):
    with zipfile.ZipFile("data.zip","r") as zip_ref:
        print("start unzipped data.zip")
        zip_ref.extractall()
        print("file unzipped")

# Parse the csv file and pack into pickle file log.p for future use
import csv
from os.path import isfile
import pickle
# if the file is not exist, parse the csv file and save it
# otherwise load it from file
if not isfile("log.p"):
    log = []
    with open('data/driving_log.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # skip the first row
        reader.__next__()
        # loop through all entry
        for line in reader:
            record = []
            record.append("data/"+line[0].strip()) # center camera
            record.append("data/"+line[1].strip()) # left camera
            record.append("data/"+line[2].strip()) # right camera
            record.append(float(line[3])) # steer angle
            log.append(record)
    # save the file
    with open('log.p', 'wb') as file:
        pickle.dump(log, file)
        print("log saved")
else:
    # load from file
    with open('log.p', 'rb') as file:
        log = pickle.load(file)
        print("log loaded")

# data augmentation helper functinos
# functions borrowed from Vivek Yadav's post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ub3zjjxme
import cv2
import numpy as np
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# %matplotlib inline
print("loading help functions")
# random shift, randomly shift the image left/right/up/down
# and adjust the angle accordingly
def random_shift(image,angle,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = 40*np.random.uniform()-40/2
    angle = angle + tr_x/trans_range*2*0.2
    rows, cols, chan = image.shape
    M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image = cv2.warpAffine(image, M, (cols,rows))
    return image, angle

# random brightness
def random_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.2+np.random.uniform() # range from 0.2 to 1.2 of original image brightness
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

# random shadow
def random_shadow(image):
    # four points to define mask area
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = 0.5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# random flip
def random_flip(image, angle):
    flip = np.random.randint(2)
    if flip:
        image = cv2.flip(image,1)
        angle = -angle
    return image, angle

# img_reader
# randomly read a image from the data set,
# then use augmentation schemes to process it.
def img_reader(record_id):
    cam_pos = np.random.randint(3)
    image_path = log[record_id][cam_pos]
    # randomly select center, left or right camera image
    # left camera shift angle 0.27, right camera shift angle -0.27
    if (cam_pos == 0):# center cam
        shift_ang = 0
    if (cam_pos == 1): # left cam
        shift_ang = 0.27
    if (cam_pos == 2): # right cam
        shift_ang = -0.27
    angle = file_path = log[record_id][3] + shift_ang
    image = mpimg.imread(image_path)
    # augmentation pipeline
    image = random_shadow(image) # add shadow
    image, angle = random_shift(image, angle, 100) # shift image
    image = random_brightness(image) # change brightness
    image, angle = random_flip(image, angle) # flip image
    return image, angle


# data generator
def data_generator(log , batch_size = 250):
    # placeholder for data
    batch_images = np.zeros((batch_size, 160, 320, 3))
    batch_angle = np.zeros(batch_size)
    while 1:
        # sample pr_threshold from (0.4 - 1)
        pr_threshold = np.random.uniform(low=0.4, high=1.0)
        # fill in batch data
        for i in range(batch_size):
            record_id = np.random.randint(len(log)) # randomly select a image
            keep_pr = 0
            while keep_pr == 0:
                image, angle = img_reader(record_id) # read the image and add augmentation
                # if the angle less than 0.1 randomly keep it according to the threshold
                if abs(angle)<0.1:
                    pr_val = np.random.uniform()
                    if pr_val > pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            batch_images[i] = image
            batch_angle[i] = angle
        yield batch_images, batch_angle

# this generator sample image from data set without any augmentation
# the generator bias to high steer angle
def val_generator(log , batch_size = 200):
    batch_images = np.zeros((batch_size, 160, 320, 3))
    batch_angle = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            angle = 0
            while abs(angle)<0.1:
                record_id = np.random.randint(len(log))
                cam_pos = 0
                image_path = log[record_id][cam_pos]
                angle = file_path = log[record_id][3]
                image = mpimg.imread(image_path)
            image, angle = random_flip(image, angle)
            batch_images[i] = image
            batch_angle[i] = angle
        yield batch_images, batch_angle
print("functions loaded")

# build model architecture
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras import backend as K
from keras.layers.core import Lambda


def resize(image_data):
    import tensorflow as tf
    return tf.image.resize_images(image_data, (66, 200))
print("building model architecture")
model = Sequential()
model.add(Cropping2D(cropping=((45, 25), (0, 0)), input_shape=(160, 320, 3), name="cropping"))
model.add(Lambda(lambda x: x/127.5 - 1.0, name="normalize"))
model.add(Lambda(resize, name="resize"))
model.add(Convolution2D(3, 1, 1, init='he_normal', border_mode='valid')) # layer to learn "best" color space
model.add(Convolution2D(24, 5, 5, activation='elu', init='he_normal', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='elu', init='he_normal', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='elu', init='he_normal', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='elu', init='he_normal', border_mode='valid', subsample=(1,1)))
model.add(Convolution2D(64, 3, 3, activation='elu', init='he_normal', border_mode='valid', subsample=(1,1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu', init='he_normal'))
model.add(Dense(50, activation='elu', init='he_normal'))
model.add(Dense(10, activation='elu', init='he_normal'))
model.add(Dense(1))
model.summary()

# model Optimization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from shutil import rmtree
from os import makedirs
from os.path import exists

# remove the old saved model and create new model folder
if exists("./model"):
    rmtree("./model")
    print("removed old model folder")
makedirs("./model")
print("created new model folder")

# save the model architecture
import json
from keras.models import model_from_json
# model.save_weights("model.h5", True)
with open("model/model.json","w") as file:
    json.dump(json.loads(model.to_json()), file)
print("model.json saved")

# open log file
# the log file stores information about camera image path and steering angle
with open('log.p', 'rb') as file:
    log = pickle.load(file)
    print("log loaded")

# create call backs to store weights after each epcoh
filepath = "model/model.{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath)
callback_list = [checkpoint]

# optimize the model
# Adam optimizer of learning rate 0.0001
# batch size = 400
# train 100 epoches and 20000 samples each epoch
print("start training process")
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mse')
gen = data_generator(log, batch_size=400)
val_gen = val_generator(log)
history = model.fit_generator(generator=gen, samples_per_epoch=20000, nb_epoch=100, validation_data=val_gen, nb_val_samples = 3000, verbose=1, callbacks=callback_list)
print("model weight saved")
