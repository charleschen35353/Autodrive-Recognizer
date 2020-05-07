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
home = str(Path.home())

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
#assert tf.executing_eagerly() == True

class_names = ["day","night"]
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHN = 3
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


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
                
class Dataloader:
    def __init__(self, data_path,  batch_size = 16):
        
       	train_imgen = keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,\
										width_shift_range = 0.35, height_shift_range = 0.35,\
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



def DayNightClassifier(): 
    
    
    input_img = layers.Input(shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHN), dtype = 'float32', name = "input_img" )
    x = layers.Conv2D(16, 3, strides=(2, 2), name = "conv1",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(input_img)
    x = layers.Conv2D(32, 3, strides=(2, 2), name = "conv2",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, strides=(2, 2), name = "conv3",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv4",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, strides=(1, 1), name = "conv5",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(2, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    

    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model

#create DNN model
data_path = "/home/charles/dataset/day_night_dataset/"
dl = Dataloader(data_path)
train_generator, val_generator = dl.load_dl()

lr = 7e-4
batch_size = 16
trainset_size = 2429
valset_size = 150
epochs = 200
checkpoint_path = "./weights_05/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model =  DayNightClassifier()
#model.load_weights("./weights_04/cp.ckpt")
keras.utils.plot_model(model, 'day_night_classification.png')



model.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = val_generator,
                                validation_steps = valset_size/batch_size,
                                use_multiprocessing = False,
                                shuffle=True,
                                verbose = 1,
                                callbacks=[cp_callback])
