import cv2
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width=width
        self.height=height
        self.inter=inter
    def preprocess(self,image):
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
    
#%%
if __name__=='__main__':  
    im=cv2.imread('/home/mohsen/MosquitoV1/AMImage/aedes/20190724_144616.jpg')
    s1=SimplePreprocessor(200,400)
    im3=s1.preprocess(im)
    im2=cv2.resize(im,(200,400))
    cv2.imshow("wom",im2)
    cv2.waitKey()
    cv2.destroyAllWindows()