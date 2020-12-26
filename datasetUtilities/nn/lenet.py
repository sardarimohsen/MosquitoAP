from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class LeNet:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        inputShape=(height,width,depth)
        #first layer
        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #second layer
        model.add(Conv2D(50,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #fc=>Relu
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        
        #softmax layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model        
                  