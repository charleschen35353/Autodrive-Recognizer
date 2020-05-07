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

class_names = ["cloudy", "rainy" ,"sunny"]
model_rain_path = "./models/model_rain3.pb" # ouputs: 0cloudy / 1rainy / 2sunny
model_sun_path = "./models/model_sun.pb" # outputs: 0cloudy / 1sunny
alpha = 0
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def predict_single_image(models, image_pyramid):
    sess_rain, sess_sun = models
    rain_output_tensor = sess_rain.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions_rain = sess_rain.run(rain_output_tensor, {'import/input_img:0':image_pyramid[0]})[0]

    sun_output_tensor = sess_sun.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions_sun = sess_sun.run(sun_output_tensor, {'import/input_img:0':image_pyramid[1]})[0]
    predictions_sun = np.array([predictions_sun[0]/2, predictions_sun[0]/2, predictions_sun[1]])
    predictions = (alpha*predictions_sun + (1-alpha)*predictions_rain)/ np.sum((alpha*predictions_sun + (1-alpha)*predictions_rain))
    pred = np.argmax(predictions, axis = -1)
    color = class_names[pred]
    return color

def load_test_image(img_path):
    img = pltimg.imread(img_path)
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)
    else:
        img = img[:,:,:3]
    image_pyramid = []
    for res in [512,256]:
        im = cv2.resize(img, dsize=(res,res), interpolation=cv2.INTER_CUBIC)
        im = im*1.0/255.0 # np image
        im = np.expand_dims(im, axis = 0) #np.resize(img, [1,256, 256, 3])
        image_pyramid.append(im)
    return image_pyramid


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

models_dir = "./models/"
with tf.gfile.GFile(model_rain_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph1:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(model_sun_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph2:
    tf.import_graph_def(graph_def)

models = [tf.Session(graph=graph1), tf.Session(graph=graph2)]   

print("models loaded.")

confusion = np.zeros(shape = (len(class_names), len(class_names)), dtype = np.int32)
total, wrong = 0,0
for i, color in zip(range(len(class_names)),class_names):
    d = os.path.join(args.target_ds, color)+"/*"
    img_paths = sorted(glob.glob(d))
    for img_path in img_paths:
        img = load_test_image(img_path)
        result = predict_single_image(models, img)
        if result != color:
            wrong += 1
            print("Wrong. The output of image from {} is {}. GT: {}".format(img_path, result, color) )
        confusion[i][class_names.index(result)] += 1
        total+=1


df = pd.DataFrame(confusion, index=["GT_" + x for x in class_names], columns=class_names)
print(df)
print("")
print("Accuracy on test data: {:4f}%".format( 100*(1.0 - (wrong / total*1.0))  ))
    
