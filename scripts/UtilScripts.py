from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from sklearn.utils import shuffle
import numpy as np
import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

def img_to_arr(p):
    with image.load_img(p) as img:
        img = image.img_to_array(img)
    return img


def img_to_arr_WMarking(p,Lnet,LaneNetRes):
    with image.load_img(p) as img:
        img = image.img_to_array(img)
    ores = img.shape[0:2]
    img = image.smart_resize(img, LaneNetRes)
    img = img[None, ...]
    imgm = Lnet(img)
    lab = np.argmax(imgm[0, :, :, :], axis=-1)
    lab = lab.astype('uint8')
    imp = img[0, :, :, :].astype('uint8')
    a = Image.fromarray(imp)
    mask = Image.fromarray(lab * 255, 'L')
    ba = np.zeros(imp.shape).astype('uint8')
    ba[:, :, 1] = lab * 255
    b = Image.fromarray(ba)
    a.paste(b, (0, 0), mask=mask)
    aa=np.array(a.resize((ores[1],ores[0])))
    return aa.astype('float32')



def normalize(img):
    img[:, :, 0] -= 94.9449
    img[:, :, 0] /= 58.6121

    img[:, :, 1] -= 103.599
    img[:, :, 1] /= 61.6239

    img[:, :, 2] -= 92.9077
    img[:, :, 2] /= 68.66

    return img

def get_model_wlabel1ch(LabModel,input_shape=(400,400)):
    #mark Labmodel to be untrainable
    LabModel.trainable=False

    #handle the input
    inputraw = tf.keras.Input(shape=(None, None, 3))
    inputs = layers.Resizing(400, 400)(inputraw)

    #get lenmarkings
    x1 = LabModel(inputs)
    x2 = tf.keras.backend.argmax(x1, axis=-1)
    x3 = layers.Reshape((400, 400, 1))(x2)
    x4=tf.cast(x3,'float32')

    #merge to one picure with labels in green channel
    #fn = tf.stack([red, green, blue,x2], axis=3)
    fn=x4
    fnr = layers.Reshape((400, 400, 1))(fn)
    #pilot net part
    pn1= Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(fnr)
    pn1a=BatchNormalization(axis=1)(pn1)
    pn2= Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn1a)
    pn2a=BatchNormalization(axis=1)(pn2)
    pn3= Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn2a)
    pn3a=BatchNormalization(axis=1)(pn3)
    pn4= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn3a)
    pn4a=BatchNormalization(axis=1)(pn4)
    pn4b= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn4a)
    pn4ba=BatchNormalization(axis=1)(pn4b)
    pn5=Flatten()(pn4ba)
    pn6=Dense(100, activation='relu')(pn5)
    pn6a=BatchNormalization()(pn6)
    pn7=Dense(50, activation='relu')(pn6a)
    pn7a=BatchNormalization()(pn7)
    pn8=Dense(10, activation='relu')(pn7a)
    pn8a=BatchNormalization()(pn8)
    out=Dense(1)(pn8)



    model = tf.keras.Model(inputraw, out)
    model.build(input_shape=input_shape)
    return model

def get_model_wlabel4ch(LabModel,input_shape=(400,400)):
    #mark Labmodel to be untrainable
    LabModel.trainable=False

    #handle the input
    inputraw = tf.keras.Input(shape=(None, None, 3))
    inputs = layers.Resizing(400, 400)(inputraw)

    #get lenmarkings
    x1 = LabModel(inputs)
    x2 = tf.keras.backend.argmax(x1, axis=-1)
    x3 = layers.Reshape((400, 400, 1))(x2)
    mask = tf.cast(tf.math.logical_not(tf.cast(x3, 'bool')), 'float32')
    x3multi = x3 * 255
    x3multi_cast = tf.cast(x3multi, 'float32')
    inputsSplit = tf.unstack(inputs, axis=3)
    red = layers.Reshape((400, 400, 1))(inputsSplit[0])
    green = layers.Reshape((400, 400, 1))(inputsSplit[1])
    blue = layers.Reshape((400, 400, 1))(inputsSplit[2])
    greenwithll = tf.add(tf.multiply(green, mask), x3multi_cast)

    #merge to one picure with labels in green channel
    x4=tf.cast(x3,'float32')
    fn = tf.stack([red, green, blue,x4], axis=3)
    fnr = layers.Reshape((400, 400, 4))(fn)
    #pilot net part
    pn1= Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(fnr)
    pn1a=BatchNormalization(axis=1)(pn1)
    pn2= Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn1a)
    pn2a=BatchNormalization(axis=1)(pn2)
    pn3= Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn2a)
    pn3a=BatchNormalization(axis=1)(pn3)
    pn4= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn3a)
    pn4a=BatchNormalization(axis=1)(pn4)
    pn4b= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn4a)
    pn4ba=BatchNormalization(axis=1)(pn4b)
    pn5=Flatten()(pn4ba)
    pn6=Dense(100, activation='relu')(pn5)
    pn6a=BatchNormalization()(pn6)
    pn7=Dense(50, activation='relu')(pn6a)
    pn7a=BatchNormalization()(pn7)
    pn8=Dense(10, activation='relu')(pn7a)
    pn8a=BatchNormalization()(pn8)
    out=Dense(1)(pn8)



    model = tf.keras.Model(inputraw, out)
    model.build(input_shape=input_shape)
    return model
def get_model_wlabel(LabModel,input_shape=(400,400)):
    #mark Labmodel to be untrainable
    LabModel.trainable=False

    #handle the input
    inputraw = tf.keras.Input(shape=(None, None, 3))
    inputs = layers.Resizing(400, 400)(inputraw)

    #get lenmarkings
    x1 = LabModel(inputs)
    x2 = tf.keras.backend.argmax(x1, axis=-1)
    x3 = layers.Reshape((400, 400, 1))(x2)
    mask = tf.cast(tf.math.logical_not(tf.cast(x3, 'bool')), 'float32')
    x3multi = x3 * 255
    x3multi_cast = tf.cast(x3multi, 'float32')
    inputsSplit = tf.unstack(inputs, axis=3)
    red = layers.Reshape((400, 400, 1))(inputsSplit[0])
    green = layers.Reshape((400, 400, 1))(inputsSplit[1])
    blue = layers.Reshape((400, 400, 1))(inputsSplit[2])
    greenwithll = tf.add(tf.multiply(green, mask), x3multi_cast)

    #merge to one picure with labels in green channel
    fn = tf.stack([red, greenwithll, blue], axis=3)
    fnr = layers.Reshape((400, 400, 3))(fn)
    #pilot net part
    pn1= Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(fnr)
    pn1a=BatchNormalization(axis=1)(pn1)
    pn2= Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn1a)
    pn2a=BatchNormalization(axis=1)(pn2)
    pn3= Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)(pn2a)
    pn3a=BatchNormalization(axis=1)(pn3)
    pn4= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn3a)
    pn4a=BatchNormalization(axis=1)(pn4)
    pn4b= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)(pn4a)
    pn4ba=BatchNormalization(axis=1)(pn4b)
    pn5=Flatten()(pn4ba)
    pn6=Dense(100, activation='relu')(pn5)
    pn6a=BatchNormalization()(pn6)
    pn7=Dense(50, activation='relu')(pn6a)
    pn7a=BatchNormalization()(pn7)
    pn8=Dense(10, activation='relu')(pn7a)
    pn8a=BatchNormalization()(pn8)
    out=Dense(1)(pn8)



    model = tf.keras.Model(inputraw, out)
    model.build(input_shape=input_shape)
    return model


# define PilotNet model, with batch normalization included.
def get_model(input_shape):
    model = Sequential([
        tf.keras.Input(shape=(None, None, 3)),
        layers.Resizing(width=input_shape[0],height = input_shape[1]),
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=1),
        Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    model.build(input_shape=input_shape)
    return model


def generator(df, batch_size, img_shape, should_shuffle,img_front_dir_path,OUTPUT_NORMALIZATION):
    # shuffle dataframe for each epoch
    if should_shuffle:
        df = shuffle(df)

    img_list = df['front']
    wheel_axis = df['wheel-axis']

    # create empty batch
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))

    index = 0
    while True:
        for i in range(batch_size):
            img_name = img_list[index]
            arr = img_to_arr(os.path.join(img_front_dir_path, img_name))

            batch_img[i] = normalize(arr)
            batch_label[i] = wheel_axis[index] / OUTPUT_NORMALIZATION

            index += 1
            if index == len(img_list):
                index = 0

        yield batch_img, batch_label


def generator_with_marking(df, batch_size, img_shape, should_shuffle,img_front_dir_path,OUTPUT_NORMALIZATION,Lnet,LaneNetRes):
    # shuffle dataframe for each epoch
    if should_shuffle:
        df = shuffle(df)

    img_list = df['front']
    wheel_axis = df['wheel-axis']

    # create empty batch
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))

    index = 0
    while True:
        for i in range(batch_size):
            img_name = img_list[index]
            arr = img_to_arr_WMarking(os.path.join(img_front_dir_path, img_name),Lnet=Lnet,LaneNetRes=LaneNetRes)

            batch_img[i] = normalize(arr)
            batch_label[i] = wheel_axis[index] / OUTPUT_NORMALIZATION

            index += 1
            if index == len(img_list):
                index = 0

        yield batch_img, batch_label

def get_angle(predict,OUTPUT_NORMALIZATION=655.35):
    angle = predict[0][0]
    angle *= OUTPUT_NORMALIZATION

    return int(angle)
