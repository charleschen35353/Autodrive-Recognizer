import glob
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import time
import argparse
import logging
import matplotlib.image as pltimg
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)

#python demo_model.py "/home/pacowong/research/datasets/hkfi_report_test_sets/test_set_1/damage/*.jpg"



color_classes = ["black", "blue", "brown", "gold", "green", "orange", "purple", "red", "silver", "white"]
model_path = "./models/simple_finetuned_83.pb"


def get_model(model_path, sess):
    f = gfile.FastGFile(model_path, 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)
    return sess

def predict_single_image(sess, image):
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input:0':image})
    pred = np.argmax(predictions, axis = -1)
    color = color_classes[pred[0]]
    return color
   
def predict_single_image_with_hist(sess, image, hist):
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image, 'import/input_hist:0':hist})
    pred = np.argmax(predictions, axis = -1)
    color = color_classes[pred[0]]
    return color

def load_test_image(img_path):
    img = pltimg.imread(img_path)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = img*1.0/255.0 # np image
    img = np.expand_dims(img, axis = 0) #np.resize(img, [1,256, 256, 3])
    return img

def get_hist_128(img):
    chans = cv2.split(img.astype(np.uint8))
    colors = ("r", "b", "g")
    features = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [128], [0.0, 255.0])
        features.extend(hist)
    features = np.squeeze(np.array(features))
    data = np.array(features)/255.0
    return data

parser = argparse.ArgumentParser()
'''
parser.add_argument("input_img_dir")
args = parser.parse_args()
print(args.input_img_dir)

sess = tf.Session()
sess = get_model(model_path,sess)
print("model loaded.")

image_paths = sorted(glob.glob(args.input_img_dir)) #'*.jpg'

for img_path in image_paths:
    img = load_test_image(img_path)
    #print("Image loaded.")
    #print("Prediction begins...")
    result = predict_single_image(sess, img)
    print("The output of image from {} is {}".format(img_path, result) )
'''

parser.add_argument("target_ds")
args = parser.parse_args()
sess = tf.Session()
sess = get_model(model_path,sess)
print("model loaded.")

confusion = np.zeros(shape = (len(color_classes), len(color_classes)), dtype = np.int32)

for i, color in zip(range(len(color_classes)),color_classes):
    d = os.path.join(args.target_ds, color)+"/*.jpg"
    img_paths = sorted(glob.glob(d))
    for img_path in img_paths:
        img = load_test_image(img_path)
        result = predict_single_image(sess, img)
        print("The output of image from {} is {}. GT: {}".format(img_path, result, color) )
        confusion[i][color_classes.index(result)] += 1


df = pd.DataFrame(confusion, index=["GT_" + x for x in color_classes], columns=color_classes)
print(df)
    
