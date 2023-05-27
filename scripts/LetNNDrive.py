import sys
sys.path.append('../')


from europilot.screen import stream_local_game_screen
from europilot.screen import Box
#from europilot.joystick import LinuxVirtualJoystick
from europilot.joystickF430 import LinuxVirtualJoystick
from UtilScripts import normalize,get_angle


import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image
from time import time

import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
TESTRUN=0

parent_path = os.path.dirname(os.getcwd())
model_path = os.path.join(parent_path, 'model')

# multiply by constant to undo normalization
OUTPUT_NORMALIZATION = 655.35

# limit GPU memory usage
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

model = load_model(os.path.join(model_path, 'v1LanePilotNet_v3_r2_2epoch_008-valloss_40.09186-vacc_0.14660.hdf5'))

front_coord = (289, 167, 851, 508)


if TESTRUN:
    sample_img = Image.open(os.path.join('../sample/img/raw', '9d0c3c2b_2017_07_27_14_55_08_16.jpg')).convert('RGB')

    sample_img_front = sample_img.crop(front_coord)
    plt.imshow(sample_img_front)
    plt.show()

    sample_arr = image.img_to_array(sample_img_front)
    sample_arr = np.reshape(sample_arr, (1,) + sample_arr.shape)

    model.predict(sample_arr, batch_size = 1)
    start = time()
    for i in range(100):
        model.predict(sample_arr, batch_size = 1)
    end = time()

    fps = 100. / (end - start)
    print("fps: %f" % fps)

box = Box(10,64,1034,832)
joy = LinuxVirtualJoystick()

print("run the game")
streamer = stream_local_game_screen(box=box, default_fps=60)
image_data = next(streamer)
im = Image.fromarray(image_data)
img_front = im.crop(front_coord)
plt.imshow(img_front)
plt.show()
while True:
    image_data = next(streamer)
    im = Image.fromarray(image_data)
    img_front = im.crop(front_coord)

    arr = image.img_to_array(img_front)
    arr = normalize(arr)
    arr = np.reshape(arr, (1,) + arr.shape)

    angle = get_angle(model.predict(arr, batch_size=1))
    print(angle)
    joy.emit(angle)
print("end")