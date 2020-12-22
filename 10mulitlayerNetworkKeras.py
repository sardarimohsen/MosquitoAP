
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
import cv2
from tensorflow.keras.models import Sequential
import numpy as np
print('OK')
#%%
imagePaths=list(paths.list_images
                ('/home/mohsen/MosquitoV1/datasetUtilities/datasetMos'))
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(dataOrig,labelsOrig)=sdl.load(imagePaths,verbose=10)
#%%
#scale the pixels value to the range [0 1.0]
data=dataOrig/255.0
lb=LabelBinarizer()
labels=lb.fit_transform(labelsOrig)
print(labels)
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25)
#%%
cv2.imshow('test',data[4])
cv2.waitKey()
cv2.destroyAllWindows()
#%%
model=Sequential()
model.add(keras.layers.Dense(256,input_shape=(32*32*3,),activation='sigmoid'))
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(5, activation='softmax'))
sgd=SGD(0.01)
model.compile(loss='categorical_crossentropy')
#%%
print(trainX.shape)
tmp=np.reshape(trainX,[trainX.shape[0],32*32*3])
H=model.fit(np.reshape(trainX,[trainX.shape[0],32*32*3]),
            trainY,
            validation_data=(np.reshape(testX,[testX.shape[0],32*32*3]),testY),
            epochs=100,batch_size=64)
#%%
tmp=np.reshape(testX,[testX.shape[0],32*32*3])
prediction=model.predict(tmp,batch_size=64)
print(classification_report(testY.argmax(axis=1),prediction.argmax(axis=1)
                            ,target_names=[str(x) for x in lb.classes_]))
#%%
print(H.history.keys())
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100),H.history["val_loss"],label='test_loss')
plt.plot(np.arange(0,100),H.history["loss"],label='training_loss')

