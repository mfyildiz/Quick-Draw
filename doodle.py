import tensorflow as tf
tf.test.gpu_device_name()

import numpy as np 
import pandas as pd
from glob import glob
from tqdm import tqdm 
import re
import ast
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import matplotlib.style as style

class_paths=glob('*.csv')
print(class_paths)
print(len(class_paths))

def draw(strokes):
  image=Image.new("P",(255,255),color=255)
  image_last=ImageDraw.Draw(image)
  for stroke in ast.literal_eval(strokes):
    for i in range(len(stroke[0])-1):
      image_last.line([stroke[0][i],stroke[0][i+1],stroke[1][i],stroke[1][i+1]],fill=0,width=5)
  image=image.resize((64,64))
  return np.array(image)/255

from dask import bag
import dask.array
trainall=[]
labelall=[]
for a,i in enumerate(tqdm(class_paths[:])):      
  imc=500
  train=pd.read_csv(i,usecols=['word','drawing','recognized'],nrows=imc*5//4)
  train=train[train.recognized==True].head(imc)
  ibag=bag.from_sequence(train.drawing.values).map(draw)
  traina=np.array(ibag.compute())
  traina=np.reshape(traina,(500,-1))
  label=np.full((train.shape[0],1),a)
  traina=np.concatenate((label,traina),axis=1)
  trainall.append(traina)
  
trainall=np.array([trainall.pop() for b in np.arange()])
trainall=trainall.reshape((-1,(64*64+1)))

del train
del traina
del label

print(trainall.shape)
print(len(trainall))
print(trainall[0].shape)
print(trainall[0])

sinir=int(0.2*trainall.shape[0])
np.random.shuffle(trainall)
Xtrain=trainall[sinir: ,1:]
Ytrain=trainall[sinir: ,0]
Xval=trainall[0:sinir,1:]
Yval=trainall[0:sinir,0]
print(Xtrain.shape)
print(Ytrain.shape)
print(Yval.shape)
del trainll

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

Ytrain=keras.utils.to_categorical(Ytrain,340)
Xtrain=Xtrain.reshape(Xtrain.shape[0],64,64,1)
Yval=keras.utils.to_categorical(Yval,340)
Xval=Xval.reshape(Xval.shape[0],64,64,1)
print(Xtrain.shape,"\n",Ytrain.shape,"\n",Xval.shape,"\n",Yval.shape)

def top_3_accuracy(x,y): 
    t3 = top_k_categorical_accuracy(x,y, 3)
    return t3

model = Sequential()
model.add(Conv2D(16, (3,3), padding = 'same',activation='relu',input_shape=(64,64,1)))
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), padding = 'same',activation='relu'))
model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), padding = 'same',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Dense(340, activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_3_accuracy])

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)
path="weights.best.hdf5"
checkpoint=ModelCheckpoint(path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [reduceLROnPlat, checkpoint]

model.fit(x=Xtrain, y=Ytrain, batch_size = 32, epochs = 30, validation_data = (Xval, Yval), callbacks = callbacks, verbose = 1)

