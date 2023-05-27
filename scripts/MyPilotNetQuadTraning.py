import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow
import keras
from sklearn.utils import shuffle
#for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done #run in shell to solve (-1) warning
#print(tf.test.is_gpu_available()) #show gpus
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from UtilScripts import *


LaneNetPath=r'/LaneLineLableModels/run_allepoch-8935-val_loss-0.0598-val_acc-0.9794.hdf5'
LaneNetRes=(400,400)
LaneMarkerModel=load_model(os.path.dirname(os.getcwd())+LaneNetPath)
LaneMarkerModel.trainable=False

USECPU=False ############################################################################################################

if USECPU:
    tf.config.set_visible_devices([], 'GPU')
EpochStop=500
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

# divide by a constant to bound output to [0,100]
OUTPUT_NORMALIZATION = 655.35

df_train = pd.read_csv(os.path.join(data_path, train_file))
print("%d rows" % df_train.shape[0])
df_train.head(3)

df_val = pd.read_csv(os.path.join(data_path, valid_file))
print("%d rows" % df_val.shape[0])
df_val.head(3)

df_train=df_train
df_val=df_val

input_shape = img_to_arr(os.path.join(img_front_dir_path, df_train['front'][0])).shape
batch_size = 15#200#160
train_steps = (df_train.shape[0] / batch_size) + 1
val_steps = (df_val.shape[0] / batch_size) + 1

print("input_shape: %s, batch_size: %d, train_steps: %d, val_steps: %d" %
      (input_shape, batch_size, train_steps, val_steps))

if NEWMODEL:
    model1ch = get_model_wlabel1ch(LaneMarkerModel)
    model4ch = get_model_wlabel4ch(LaneMarkerModel)
    model3ch = get_model_wlabel(LaneMarkerModel)
    modelori=get_model(input_shape=(400,400))

    sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)

    model1ch.compile(optimizer=sgd, loss="mse",metrics='accuracy')
    model4ch.compile(optimizer=sgd, loss="mse", metrics='accuracy')
    model3ch.compile(optimizer=sgd, loss="mse", metrics='accuracy')
    modelori.compile(optimizer=sgd, loss="mse", metrics='accuracy')

    model1ch.summary()
    model4ch.summary()
    model3ch.summary()
    modelori.summary()


train_batch = generator(df_train, batch_size, input_shape, True,img_front_dir_path=img_front_dir_path,OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION)
val_batch = generator(df_val, batch_size, input_shape, False,img_front_dir_path=img_front_dir_path,OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION)

cur_model1ch = cur_file + '1chPilotNet'
cur_model4ch = cur_file + '4chPilotNet'
cur_model3ch = cur_file + '3chPilotNet'
cur_modelori = cur_file + 'oriPilotNet'


csv_logger1ch = CSVLogger(os.path.join(log_path, cur_model1ch + '.log'),append=True)
csv_logger4ch = CSVLogger(os.path.join(log_path, cur_model4ch + '.log'),append=True)
csv_logger3ch = CSVLogger(os.path.join(log_path, cur_model3ch + '.log'),append=True)
csv_loggerori = CSVLogger(os.path.join(log_path, cur_modelori + '.log'),append=True)


model_file_name1ch= os.path.join(model_path, cur_model1ch + 'epoch_{epoch:03d}-valloss_{val_loss:.5f}-vacc_{val_accuracy:.5f}.hdf5')
model_file_name4ch= os.path.join(model_path, cur_model4ch + 'epoch_{epoch:03d}-valloss_{val_loss:.5f}-vacc_{val_accuracy:.5f}.hdf5')
model_file_name3ch= os.path.join(model_path, cur_model3ch + 'epoch_{epoch:03d}-valloss_{val_loss:.5f}-vacc_{val_accuracy:.5f}.hdf5')
model_file_nameori= os.path.join(model_path, cur_modelori + 'epoch_{epoch:03d}-valloss_{val_loss:.5f}-vacc_{val_accuracy:.5f}.hdf5')

checkpoint1ch = ModelCheckpoint(model_file_name1ch, verbose=0, save_best_only=False,save_weights_only=False,monitor='val_loss',mode='min')
checkpoint4ch = ModelCheckpoint(model_file_name4ch, verbose=0, save_best_only=False,save_weights_only=False,monitor='val_loss',mode='min')
checkpoint3ch = ModelCheckpoint(model_file_name3ch, verbose=0, save_best_only=False,save_weights_only=False,monitor='val_loss',mode='min')
checkpointori = ModelCheckpoint(model_file_nameori, verbose=0, save_best_only=False,save_weights_only=False,monitor='val_loss',mode='min')


epochbypass=1
for i in range(0,EpochStop):
    print("1chPilot")
    model1ch.fit_generator(train_batch,train_steps,epochs=epochbypass*(i+1),verbose=1,callbacks=[csv_logger1ch, checkpoint1ch],validation_data=val_batch,validation_steps=val_steps,initial_epoch=epochbypass*i)
    print("4chPilot")
    model4ch.fit_generator(train_batch,train_steps,epochs=epochbypass*(i+1),verbose=1,callbacks=[csv_logger4ch, checkpoint4ch],validation_data=val_batch,validation_steps=val_steps,initial_epoch=epochbypass*i)
    print("3chPilot")
    model3ch.fit_generator(train_batch,train_steps,epochs=epochbypass*(i+1),verbose=1,callbacks=[csv_logger3ch, checkpoint3ch],validation_data=val_batch,validation_steps=val_steps,initial_epoch=epochbypass*i)
    print("oriPilot")
    modelori.fit_generator(train_batch,train_steps,epochs=epochbypass*(i+1),verbose=1,callbacks=[csv_loggerori, checkpointori],validation_data=val_batch,validation_steps=val_steps,initial_epoch=epochbypass*i)

print("end")