from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from tensorflow.keras import layers
from tensorflow import keras
from pathlib import Path
from scipy import ndimage
import tensorflow.keras.backend as K
home = str(Path.home())

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
#assert tf.executing_eagerly() == True

class_names = ["cloudy", "rainy" ,"sunny"]
IMG_HEIGHT = 256 #IMG_HEIGHT = 128
IMG_WIDTH = 256 #IMG_WIDTH = 128

IMG_CHN = 3
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def sharpen_edge(img):
    blurred_f = ndimage.gaussian_filter(img, 3)

    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 30
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
'''
def generate_generator(generator, path, batch_size = 16, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

        gen = generator.flow_from_directory(path,
                                            classes = class_names,
                                            target_size = (img_height,img_width),
                                            batch_size = batch_size,
                                            shuffle=True, 
                                            seed=7)
        while True:
            X,y = gen.next()    
            yield X, y #Yield both images and their mutual label
'''              
def generate_generator(generator, path, batch_size = 16, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

        gen = generator.flow_from_directory(path,
                                            classes = class_names,
                                            target_size = (img_height,img_width),
                                            batch_size = batch_size,
                                            shuffle=True, 
                                            seed=7)
        while True:
            X,y = gen.next()
            #i = 0
            '''
            sobel_X = []
            for img in X:   
                img = np.exp(img*2)
                img = ( img - np.min(img) ) / ( np.max(img) - np.min(img) ) *255.0 #increase contrast
                sobel_img = ndimage.sobel(rgb2gray(img*255.0) )
                sobel_X.append(sobel_img / 255.0)
                #cv2.imwrite("test_x{}.jpg".format(i),sobel_X)
                #cv2.imwrite("origin_x{}.jpg".format(i),img*255.0)
                #i+=1
            sobel_X = np.array(sobel_X)
            sobel_X = np.expand_dims(sobel_X, axis = -1)
            X = np.concatenate([X,sobel_X], axis = -1)
            '''
            yield X, y #Yield both images and their mutual label
             
class Dataloader:
    def __init__(self, data_path,  batch_size = 16):
        
       	train_imgen = keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,\
										horizontal_flip = True, rescale = 1/255.0)

        test_imgen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)

        self.train_generator = generate_generator(train_imgen,
                                               path = str(data_path) + "train/",
                                               batch_size=batch_size)       

        self.val_generator = generate_generator(test_imgen,
                                              path = str(data_path)+ "val/",
                                              batch_size=batch_size)              

        
    def load_image(self, val = False):
        if val:
            return next(self.val_generator)
        else:
            return next(self.train_generator)
    
    def load_dl(self):
        return [self.train_generator, self.val_generator]

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def WeatherSceneClassifier(): 
    
    
    input_img = layers.Input(shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHN), dtype = 'float32', name = "input_img" )
    x = layers.Conv2D(16, 3, strides=(2, 2), name = "conv1",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(input_img)
    conv2_out = layers.Conv2D(32, 3, strides=(2, 2), name = "conv2",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.MaxPool2D()(conv2_out)
    x = layers.Conv2D(64, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv5",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform')(x)
    lf = layers.MaxPool2D(pool_size=(4, 4))(conv2_out)
    x = layers.Concatenate(axis = -1)([x,lf])
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(3, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    

    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss = weighted_categorical_crossentropy([1,1,1]),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model

'''
def WeatherSceneClassifier(): 
    
    
    input_img = layers.Input(shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHN), dtype = 'float32', name = "input_img" )
    x = layers.Conv2D(16, 3, strides=(2, 2), name = "conv1",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(input_img)
    x = layers.Conv2D(32, 3, strides=(2, 2), name = "conv2",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, strides=(1, 1), name = "conv3",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv4",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv5",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(3, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    

    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss = weighted_categorical_crossentropy([1,1,1.5]),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model
'''
#create DNN model
data_path = "/home/charles/dataset/weather/"
dl = Dataloader(data_path)
train_generator, val_generator = dl.load_dl()

lr = 7e-4
batch_size = 16
trainset_size = 3967
valset_size = 100
epochs = 500
checkpoint_path = "./weights_filtered/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model = WeatherSceneClassifier()
#model.load_weights("./weights_filtered/cp.ckpt")
keras.utils.plot_model(model, 'weather_classification.png')
	


model.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = val_generator,
                                validation_steps = valset_size/batch_size,
                                use_multiprocessing = False,
                                shuffle=True,
                                verbose = 1,
                                callbacks=[cp_callback])
