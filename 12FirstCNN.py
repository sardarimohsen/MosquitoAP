#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:38:00 2020

@author: mohsen
"""
#import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from datasetUtilities.datasetMos import SimpleDatasetLoader
from datasetUtilities.preprocessing import SimplePreprocessor
from datasetUtilities.preprocessing import ImageToArrayPreprocessor
from datasetUtilities.nn import ShallowNet
import cv2
from tensorflow.keras.models import Sequential
import numpy as np
print('OK')
#%%
imagePaths=list(paths.list_images
                ('/home/mohsen/MosquitoV1/datasetUtilities/datasetMos'))
sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreprocessor()
sdl=SimpleDatasetLoader(preprocessors=[sp,iap])
(dataOrig,labelsOrig)=sdl.load(imagePaths,verbose=10)
dataOrig=dataOrig.astype('float')/255.0
#%%
cv2.imshow('test',dataOrig[4])
cv2.waitKey()
cv2.destroyAllWindows()
#%%
lb=LabelBinarizer()
labels=lb.fit_transform(labelsOrig)
print(labels)
(trainX,testX,trainY,testY)=train_test_split(dataOrig,labels,test_size=0.25,random_state=42)
#%%
print('INFO compiling...')
opt=SGD(lr=0.005)
model=ShallowNet.build(width=32,height=32,depth=3,classes=5)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=500,verbose=1)
#%%
prediction=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),prediction.argmax(axis=1)
                            ,target_names=[str(x) for x in lb.classes_]))
#%%
plt.close()
print(H.history.keys())
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,500),H.history["val_accuracy"],label='test_acc')
plt.plot(np.arange(0,500),H.history["accuracy"],label='training_acc')
plt.legend()
plt.show()