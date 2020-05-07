from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
#tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow import keras
from pathlib import Path
from tensorflow.python.platform import gfile
import time
import argparse
import logging
import matplotlib.image as pltimg
import pandas as pd
import glob
tf.logging.set_verbosity(tf.logging.ERROR)


classes = ["indoor", "outdoor"]

models_dir = "./models/"
with tf.gfile.GFile(models_dir + "weights_97.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph1:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(models_dir + "weights_95.6.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph2:
    tf.import_graph_def(graph_def)

models = [tf.Session(graph=graph1), tf.Session(graph=graph2)]
 

def predict_single_image_logits(sess, image):
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    return predictions

def predict_single_image(sess, image):
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    pred = np.argmax(predictions, axis = -1)
    color = classes[pred[0]]
    return color

def predict_single_image_logits(sess, image):
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    return predictions

def load_test_image(img_path):
    img = pltimg.imread(img_path)
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)
    else:
        img = img[:,:,:3]
    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    img = img*1.0/255.0 # np image
    img = np.expand_dims(img, axis = 0) #np.resize(img, [1,256, 256, 3])
    return img

def ensemble_predict(models, image):
    logits_1 = predict_single_image_logits(models[0], image)
    logits_2 = predict_single_image_logits(models[1], image)
    logits = (logits_1 + logits_2) / 2
    pred = np.argmax(logits, axis = -1)
    color = classes[ int(pred[0] != 0)]
    return color
    
parser = argparse.ArgumentParser()
parser.add_argument("target_ds")
args = parser.parse_args()
print("Two models loaded.")

#ensemble prediction

total, wrong = 0,0
confusion = np.zeros(shape = (len(classes), len(classes)), dtype = np.int32)
for i, color in zip(range(len(classes)),classes):
    d = os.path.join(args.target_ds, color)+"/*"
    img_paths = sorted(glob.glob(d))
    for img_path in img_paths:
        img = load_test_image(img_path)
        result = ensemble_predict(models, img)
        if result != color:
            wrong += 1
            #print("Wrong. The output of image from {} is {}. GT: {}".format(img_path, result, color) )
        confusion[i][classes.index(result)] += 1
        total+=1
df = pd.DataFrame(confusion, index=["GT_" + x for x in classes], columns=classes)
print(df)
print("")
print("Ensemble model accuracy on test data: {:4f}%".format( 100*(1.0 - (wrong / total*1.0))  ))
    
