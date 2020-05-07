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

class_names = ["day", "night"]
IMG_HEIGHT =64
IMG_WIDTH = 64
IMG_CHN = 3
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

lr = 1
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
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(2, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    

    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model
class Dataloader:
    def __init__(self, data_path,  batch_size = 16):
        
        test_imgen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
        self.test_generator = generate_generator(test_imgen,
                                              path = str(data_path)+ "test/",
                                              batch_size=batch_size)            
    def load_dl(self):
        return self.test_generator
    
batch_size = 16
checkpoint_path =  "weights_05/cp.ckpt"
testset_size = 372
data_path = "/home/charles/dataset/day_night_dataset/"
dl = Dataloader(data_path)
test_generator = dl.load_dl()
model = DayNightClassifier()
model.load_weights(checkpoint_path)

loss, acc = model.evaluate_generator(test_generator,steps = testset_size/batch_size,use_multiprocessing = False)
print("Restored model, loss: {}, accuracy: {:5.2f}%".format(loss, acc*100))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


from keras import backend as K

# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

print([out.op.name for out in model.outputs])
print([out.op.name for out in model.inputs])

tf.train.write_graph(frozen_graph, "./", "model01.pb", as_text=False)
[n.name for n in tf.get_default_graph().as_graph_def().node]

print("model output to pb successful.")

