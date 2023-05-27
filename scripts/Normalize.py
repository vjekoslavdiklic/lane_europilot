import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import keras
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

def img_to_arr(p):
    with image.load_img(p) as img:
        img = image.img_to_array(img)
    return img

cur_file = 'v1'
parent_path = os.path.dirname(os.getcwd())

data_path = os.path.join(parent_path, 'scripts','data')
img_front_dir_path = os.path.join(data_path, 'img', 'front')
model_path = os.path.join(parent_path, 'model')
log_path = os.path.join(model_path, 'log')


csv_dir_path = os.path.join(data_path, 'csv', 'final')
train_file = os.path.join(csv_dir_path, cur_file + '_train.csv')
valid_file = os.path.join(csv_dir_path, cur_file + '_valid.csv')

# divide by a constant to bound input and output to [0,1]
INPUT_NORMALIZATION = 255
OUTPUT_NORMALIZATION = 65535

df_train = pd.read_csv(os.path.join(data_path, train_file))
print("%d rows" % df_train.shape[0])
df_train.head(3)

df_val = pd.read_csv(os.path.join(data_path, valid_file))
print("%d rows" % df_val.shape[0])
df_val.head(3)

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

df = shuffle(df_train)
X_train_sample = np.array([img_to_arr(os.path.join(img_front_dir_path, p)) for p in df['front'][:2500]])
for i in range(0,3):
    print(X_train_sample[:,:,:,i].mean(), X_train_sample[:,:,:,i].std())
sample_img = img_to_arr(os.path.join(img_front_dir_path, df['front'][1]))
sample_img.mean(), sample_img.std()