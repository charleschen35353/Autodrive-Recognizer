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

classes = ["indoor", "outdoor"]
model_path = "./models/weights_96.22.pb"

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
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    pred = np.argmax(predictions, axis = -1)
    color = classes[ int(pred[0] != 0)]
    return color

def load_test_image(img_path):
    img = pltimg.imread(img_path)
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)
    else:
        img = img[:,:,:3]
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = img*1.0/255.0 # np image
    img = np.expand_dims(img, axis = 0) #np.resize(img, [1,256, 256, 3])
    return img


parser = argparse.ArgumentParser()
parser.add_argument("target_ds")
args = parser.parse_args()
sess = tf.Session()
sess = get_model(model_path,sess)
print("model loaded.")

confusion = np.zeros(shape = (len(classes), len(classes)), dtype = np.int32)
total, wrong = 0,0
for i, color in zip(range(len(classes)),classes):
    d = os.path.join(args.target_ds, color)+"/*"
    img_paths = sorted(glob.glob(d))
    for img_path in img_paths:
        img = load_test_image(img_path)
        result = predict_single_image(sess, img)
        if result[:3] != color[:3]:
            wrong += 1
            print("Wrong. The output of image from {} is {}. GT: {}".format(img_path, result, color) )
        confusion[i][classes.index(result)] += 1
        total+=1


df = pd.DataFrame(confusion, index=["GT_" + x for x in classes], columns=classes)
print(df)
print("")
print("Accuracy on test data: {:4f}%".format( 100*(1.0 - (wrong / total*1.0))  ))
    
