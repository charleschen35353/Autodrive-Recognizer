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
import random
import tensorflow.keras.backend as K
from scipy import ndimage, misc
home = str(Path.home())

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
#assert tf.executing_eagerly() == True

class_names = ["cloudy", "rainy" ,"sunny"]
#class_names = ["rainy", "not-rainy"]
NUM_OF_CLASSES = len(class_names)
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHN = 3
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
RF = 1e-3
	
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def generate_generator(generator, path, batch_size = 16, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

        gen = generator.flow_from_directory(path,
                                            classes = class_names,
                                            target_size = (img_height, img_width),
                                            batch_size = batch_size,
                                            shuffle=True, 
                                            seed=7)
        while True:
            X,y = gen.next()
            '''
            X_sobel = []
            for img,lbl,i in zip(X,y, range(X.shape[0])):
                print(np.where(lbl == 1.0)[0][0])
                if True:#np.where(lbl == 1.0)[0][0] == 1:
                    kernel = np.ones((5,5),np.float32)/25
                    img = cv2.filter2D(img, -1, kernel)
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    img = cv2.filter2D(img, -1, kernel)
                    img = cv2.filter2D(img, -1, kernel)
                    edges = cv2.Canny(np.uint8(rgb2gray(img)*255), 50, 200)
                    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider, minLineLength=10, maxLineGap=250)
                    #sobelx8u = cv2.Sobel(rgb2gray(img),cv2.CV_8U,1,0,ksize=5)
                    cv2.imwrite("test{}.jpg".format(i), cv2.cvtColor(edges, cv2.COLOR_BGR2RGB) )
                    cv2.imwrite("origin{}.jpg".format(i), cv2.cvtColor(img*255, cv2.COLOR_BGR2RGB))
                    X_sobel.append(edges)
            X_sobel = np.array(X_sobel)
            '''
            yield X, y

class Dataloader:
    def __init__(self, data_path,  batch_size = 16):
        
       	train_imgen = keras.preprocessing.image.ImageDataGenerator(rotation_range = 20,\
										#width_shift_range = 0.15, height_shift_range = 0.15,\
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

def Classifier(): 
    '''
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHN)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                  include_top=False,
                                                  weights='imagenet')
    
    base_model.trainable = False
    #for i in range(2):
    #   base_model.layers[-1-i].trainable = True
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(32, activation= 'relu', name = "fc", kernel_initializer = 'glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    output = layers.Dense(NUM_OF_CLASSES, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    model = keras.Model(base_model.input, output)

    #model = model_base
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss = weighted_categorical_crossentropy([1 for _ in range(NUM_OF_CLASSES)]),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    print(model.summary())
    for l in model.layers:
        print(l.trainable)
    return model
    '''
    input_img = layers.Input(shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHN), dtype = 'float32', name = "input_img" )
    x = layers.Conv2D(16, 3, strides=(2, 2), name = "conv1",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(input_img)
    x = layers.Conv2D(32, 3, strides=(2, 2), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform',  kernel_regularizer = keras.regularizers.l2(RF))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, strides=(2, 2), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    x = layers.Conv2D(128, 3, strides=(2, 2), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv5",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv6",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(RF))(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv7",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform',  kernel_regularizer = keras.regularizers.l2(RF))(x)
    #lf = layers.MaxPool2D(pool_size=(4, 4))(conv2_out)
    #x = layers.Concatenate(axis = -1)([x,lf])
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(NUM_OF_CLASSES, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    

    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.categorical_crossentropy,#weighted_categorical_crossentropy([1, 1/0.1, 1/0.65]),
              metrics=[tf.keras.metrics.CategoricalAccuracy()] )
    
    return model
    
#create DNN model
#data_path = "/home/charles/dataset/weather_car_rain_no/"
data_path = "/home/charles/dataset/weather_car/"
dl = Dataloader(data_path)
train_generator, val_generator = dl.load_dl()

lr = 7e-4
batch_size = 16
trainset_size = 1557
valset_size = 180
epochs = 500
checkpoint_path = "./weights_ca_new/cp.ckpt"
#checkpoint_path = "./weights_car_small/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model = Classifier()
#model.load_weights("./weights_car/cp.ckpt")
keras.utils.plot_model(model, 'Net.png')
	
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=20, min_lr=1e-5)
model.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = val_generator,
                                validation_steps = valset_size/batch_size,
                                use_multiprocessing = False,
                                shuffle=True,
                                verbose = 1,
                                callbacks=[cp_callback, reduce_lr])
