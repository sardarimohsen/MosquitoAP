#import datasetMos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
from imutils import paths
from datasetUtilities.datasetMos import SimpleDatasetLoader
from datasetUtilities.preprocessing import SimplePreprocessor
print('OK')
#%%
imagePaths=list(paths.list_images
                ('/home/mohsen/MosquitoV1/datasetUtilities/datasetMos'))
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data,labels)=sdl.load(imagePaths,verbose=10)
#%%
data=data.reshape((data.shape[0],32*32*3))
print('Info: feature matrix:{:.1f}MB'.format(data.nbytes/(1024*1000.0)))
#%%
le=LabelEncoder()
labels=le.fit_transform(labels)
(trainX,testX,trainY,testY)=train_test_split(data,labels,
    test_size=0.25,random_state=42)
#%%
model=KNeighborsClassifier()
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))
#%%
