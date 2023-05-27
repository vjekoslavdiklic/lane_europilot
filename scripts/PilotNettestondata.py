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
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from UtilScripts import *

model1path=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/v1LanePilotNet_v3_r2_2epoch_300-valloss_55.71003-vacc_0.24002.hdf5'
model2path=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/v1PilotNet_v3_r2_2epoch_300-valloss_0.85940-vacc_0.26708.hdf5'
model3path=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/v1100_1chPilotNet_v3_r2_2epoch_300-valloss_229.78044-vacc_0.29819.hdf5'
model4path=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/v1100_4chPilotNet_v3_r2_2epoch_300-valloss_5.56944-vacc_0.25610.hdf5'



model1maxvalaccpath=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/MaxValAcc/v1LanePilotNet_v3_r2_2epoch_296-valloss_73.11584-vacc_0.25590.hdf5'
model2maxvalaccpath=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/MaxValAcc/v1PilotNet_v3_r2_2epoch_290-valloss_0.86838-vacc_0.27284.hdf5'
model3maxvalaccpath=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/MaxValAcc/v1100_1chPilotNet_v3_r2_2epoch_137-valloss_4602.91309-vacc_0.29844.hdf5'
model4maxvalaccpath=r'/home/vjekod/Desktop/MIPRO2023/PyScripts/PilotModels/MaxValAcc/v1100_4chPilotNet_v3_r2_2epoch_284-valloss_4.04585-vacc_0.26613.hdf5'

tf.config.set_visible_devices([], 'GPU')

# define path variables
parent_path = os.path.dirname(os.getcwd())

data_path = os.path.join(parent_path,'scripts', 'data')
img_front_dir_path = os.path.join(data_path, 'img', 'front')
model_path = os.path.join(parent_path, 'model')
log_path = os.path.join(model_path, 'log')


csv_dir_path = os.path.join(data_path, 'csv', 'final')
cur_file = 'e1'
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
df_val=df_val[0:1000]

input_shape = img_to_arr(os.path.join(img_front_dir_path, df_train['front'][0])).shape
batch_size = 100#200#160
train_steps = (df_train.shape[0] / batch_size) + 1
val_steps = (df_val.shape[0] / batch_size) + 1

print("input_shape: %s, batch_size: %d, train_steps: %d, val_steps: %d" %
      (input_shape, batch_size, train_steps, val_steps))

val_batch = generator(df_val, batch_size, input_shape, False,img_front_dir_path=img_front_dir_path,OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION)

model1=load_model(model1path)
model2=load_model(model2path)

model3=load_model(model3path)
model4=load_model(model4path)


model1a=load_model(model1maxvalaccpath)
model2a=load_model(model2maxvalaccpath)

model3a=load_model(model3maxvalaccpath)
model4a=load_model(model4maxvalaccpath)



def makeplotanalysis(model1,model2,model3,model4,val_batch,a=0,b=-1,stopafterpass=20,mtitle='epoch=300'):
    firstpass=1
    i=0
    for data,label in val_batch:
        i=i+1
        est1raw = np.array(model1(data))
        est2raw = np.array(model2(data))
        est3raw = np.array(model3(data))
        est4raw = np.array(model4(data))
        if firstpass:
            firstpass=0
            ets1 = est1raw
            ets2 = est2raw
            ets3 = est3raw
            ets4 = est4raw
            labels=label
        else:
            ets1 = np.vstack((ets1, est1raw))
            ets2 = np.vstack((ets2, est2raw))
            ets3 = np.vstack((ets3, est3raw))
            ets4 = np.vstack((ets4, est4raw))
            labels=np.vstack((labels,label))
        #print("pass")
        if stopafterpass==i:
            break
        #ets1.append((np.array(est1raw[:, 0])))
        #ets2.append((np.array(est2raw[:, 0])))
        #labels.append(((label[:,0])))
        #break


    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(labels[a:b],ets1[a:b])
    m1=m.result().numpy()

    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(labels[a:b],ets2[a:b])
    m2=m.result().numpy()

    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(labels[a:b],ets3[a:b])
    m3=m.result().numpy()

    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(labels[a:b],ets4[a:b])
    m4=m.result().numpy()



    aa = tf.keras.metrics.Accuracy()
    aa.update_state(labels[a:b],ets1[a:b])
    a1=aa.result().numpy()

    aa = tf.keras.metrics.Accuracy()
    aa.update_state(labels[a:b],ets2[a:b])
    a2=aa.result().numpy()

    aa = tf.keras.metrics.Accuracy()
    aa.update_state(labels[a:b],ets3[a:b])
    a3=aa.result().numpy()

    aa = tf.keras.metrics.Accuracy()
    aa.update_state(labels[a:b],ets4[a:b])
    a4=aa.result().numpy()

#####


    plt.figure(dpi=1000)
    plt.plot(labels[a:b])
    plt.plot(ets1[a:b])
    plt.plot(ets2[a:b])
    plt.plot(ets3[a:b])
    plt.plot(ets4[a:b])
    plt.legend(['label','lan_mse:'+str("%3.5f"%m1)+' acc:'+str("%3.5f"%a1),
                        'eur_mse:'+str("%3.5f"%m2)+' acc:'+str("%3.5f"%a2),
                        '1ch_mse:'+str("%3.5f"%m3)+' acc:'+str("%3.5f"%a3),
                        '4ch_mse:'+str("%3.5f"%m4)+' acc:'+str("%3.5f"%a4)],loc='upper right')
    plt.title(mtitle)
    plt.xlabel('frames')
    plt.ylabel("wheel turn angle")
    plt.ylim([-10,10])
    plt.show()







makeplotanalysis(model1,model2,model3,model4,val_batch,a=0,b=-1)
makeplotanalysis(model1a,model2a,model3a,model4a,val_batch,a=0,b=-1,mtitle='Max Vall Acc')


plt.figure(dpi=1000,figsize=(16,9))
plt.plot(labels[a:b])
plt.plot(ets1[a:b])
plt.plot(ets2[a:b])
plt.legend(['label','lanepilot_mse:'+str(m1*10.8),'europilot_mse:'+str(m2*10.8)])
plt.show()


print("end")