"""
Used for marking whole folder with driving footage.
"""
import glob

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
projectDir='/home/vjekod/Desktop/europilot/'

modelPath=projectDir+'/LaneLineLableModels/Oxford_Unet_CuLane_1st_50k_epoch-0474-val_loss-0.0527-val_acc-0.9807.hdf5'
pathToPicutres='scripts/data/img/front'
pathtosaveplots='scripts/data/img/front_marked'

imageslist=glob.glob(projectDir+pathToPicutres+'/*.jpg')
model=tf.keras.models.load_model(modelPath)

def overlapplot(X,Ye,savepath=False,index=0,Onlyimagereturn=False,transparancy=125,Savemode='imgonly',Targetres=None):
    if len(X.shape)==4:
        x=X[index, :, :, :]
        y=Ye[index, :, :, 0]
    else:
        x=X
        y=Ye
    a = Image.fromarray(x.astype('uint8'))
    ba = np.zeros(x.shape).astype('uint8')
    ba[:, :, 1] = y.astype('uint8') * 255
    mask = Image.fromarray(y.astype('uint8') *transparancy, 'L')
    b = Image.fromarray(ba)

    a.paste(b, (0, 0), mask=mask)
    if Targetres!=None:
        a=a.resize(Targetres)
    if Onlyimagereturn==False:
        plt.imshow(a)
        if savepath==None:
            plt.show()
        else:
            if Savemode=='imgonly':
                a.save(savepath)
            else:
                plt.savefig(savepath)
    else:
        return a
    return a


for each in imageslist:
    Currentimage = Image.open(each)
    originalres=Currentimage.size
    Currentimage = np.array(Currentimage.resize((400, 400)))
    Currentimage = Currentimage[None,...]
    Prediction=model(Currentimage)
    Prediction=np.argmax(Prediction,axis=-1)[...,None]
    CurrentPath='/'+each.split('\\')[-1]
    SavePath=CurrentPath.replace(pathToPicutres, pathtosaveplots)
    overlapplot(X=Currentimage, Ye= Prediction, savepath=SavePath, index=0, Onlyimagereturn=False, transparancy=255,Savemode='imgonly',Targetres=originalres)


