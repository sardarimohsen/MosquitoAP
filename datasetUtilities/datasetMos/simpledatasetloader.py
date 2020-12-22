import numpy as np
import cv2
import os
class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors=preprocessors
        if self.preprocessors is None:
            self.preprocessors=[]
    def load(self,imagePaths,verbose=-1):
        data=[]
        labels=[]
        for(i,imagePath) in enumerate(imagePaths):
            image=cv2.imread(imagePath)
            label=imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image=p.preprocess(image)
            data.append(image)
            labels.append(label)
            if verbose>0 and i>0 and (i+1)%verbose==0:
                print("[Info] processed {}/{}".format(i+1,len(imagePaths)))
        return (np.array(data),np.array(labels))
    
#%%
if __name__=='__main__':
    s1=SimpleDatasetLoader()
    (data,labels)=s1.load(['/home/mohsen/MosquitoV1/datasetUtilities/datasetMos/aedes/20190724_145454_hf.jpg',
             '/home/mohsen/MosquitoV1/datasetUtilities/datasetMos/aedes/20190724_145748_hf.jpg',
              '/home/mohsen/MosquitoV1/datasetUtilities/datasetMos/aedes/20190724_145748_hf.jpg']
    ,verbose=1)
    cv2.imshow('test',data[0])
    cv2.waitKey()
    cv2.destroyAllWindows()
    print('Hello')