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

USECPU=True ############################################################################################################

if USECPU:
    tf.config.set_visible_devices([], 'GPU')

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


input_shape = img_to_arr(os.path.join(img_front_dir_path, df_train['front'][0])).shape
batch_size = 200#160
train_steps = (df_train.shape[0] / batch_size) + 1
val_steps = (df_val.shape[0] / batch_size) + 1

print("input_shape: %s, batch_size: %d, train_steps: %d, val_steps: %d" %
      (input_shape, batch_size, train_steps, val_steps))

if NEWMODEL:
    model = get_model(input_shape)
    sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse")
    model.summary()
if LOADMODEL:
    model = load_model(os.path.join(model_path, 'v1-PilotNet_v3_r2-062-98.86456.h5'))


train_batch = generator(df_train, batch_size, input_shape, True,img_front_dir_path=img_front_dir_path,OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION)
val_batch = generator(df_val, batch_size, input_shape, False,img_front_dir_path=img_front_dir_path,OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION)

cur_model = cur_file + 'SW_PilotNet_v3_r2_2'
csv_logger = CSVLogger(os.path.join(log_path, cur_model + '.log'))

model_file_name= os.path.join(model_path, cur_model + '-{epoch:03d}-{val_loss:.5f}.h5')
checkpoint = ModelCheckpoint(model_file_name, verbose=0, save_best_only=True)

model.fit_generator(train_batch,
                    train_steps,
                    epochs=100,
                    verbose=1,
                    callbacks=[csv_logger, checkpoint],
                    validation_data=val_batch,
                    validation_steps=val_steps,
                    initial_epoch=0)










print("end")