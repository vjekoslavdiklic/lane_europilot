import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow
import keras
from sklearn.utils import shuffle
#for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done #run in shell to solve (-1) warning
#print(tf.test.is_gpu_available()) #show gpus
from keras.models import Sequential
from keras.models import load_model
from UtilScripts import get_model_wlabel
from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from UtilScripts import *
ETS2img1=r'/home/vjekod/Desktop/CULane_seg_label_generate/ets2testdata/3ac216af_2022_11_03_10_57_07_60_front.jpg'
testimg=tf.keras.utils.load_img(path=ETS2img1)
testimg=tf.image.resize(testimg,(400,400))
USECPU=True ############################################################################################################

#if USECPU:
#    tf.config.set_visible_devices([], 'GPU')

NEWMODEL=1
LOADMODEL=0

GpuMemLimit=20000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GpuMemLimit)])
  except RuntimeError as e:
    print(e)

# define path variables
parent_path = os.path.dirname(os.getcwd())

data_path = os.path.join(parent_path,'scripts', 'data')
img_front_dir_path = os.path.join(data_path, 'img', 'front')
model_path = os.path.join(parent_path, 'model')
log_path = os.path.join(model_path, 'log')


csv_dir_path = os.path.join(data_path, 'csv', 'final')
cur_file = 'v1'
train_file = os.path.join(csv_dir_path, cur_file + '_train.csv')
valid_file = os.path.join(csv_dir_path, cur_file + '_valid.csv')

LaneNetPath=r'/LaneLineLableModels/Oxford_Unet_CuLane_1st_50k_epoch-0474-val_loss-0.0527-val_acc-0.9807.hdf5'
LaneNetRes=(400,400)
LaneMarkerModel=load_model(os.path.dirname(os.getcwd())+LaneNetPath)
LaneMarkerModel.trainable=False


inputraw = keras.Input(shape=(None,None,3))
inputs=layers.Resizing(400,400)(inputraw)
x1=LaneMarkerModel(inputs)
x2=tf.keras.backend.argmax(x1,axis=-1)
x3=layers.Reshape((400,400,1))(x2)
mask=tf.cast(tf.math.logical_not(tf.cast(x3,'bool')),'float32')
x3multi=x3*255
x3multi_cast=tf.cast(x3multi,'float32')
inputsSplit=tf.unstack(inputs,axis=3)
red=layers.Reshape((400,400,1))(inputsSplit[0])
green=layers.Reshape((400,400,1))(inputsSplit[1])
blue=layers.Reshape((400,400,1))(inputsSplit[2])
greenwithll=tf.add(tf.multiply(green,mask),x3multi_cast)
fn=tf.stack([red,greenwithll,blue],axis=3)
fnr=layers.Reshape((400,400,3))(fn)
model = keras.Model(inputraw , fnr)



import matplotlib.pyplot as plt

xxx=model(testimg[None,...])
y=np.array(xxx[0,:,:,:])
plt.imshow(y.astype('uint8'))
plt.show()

nnm=get_model_wlabel(LaneMarkerModel)

print("waithere")