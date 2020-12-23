#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:42:38 2020

@author: mohsen
"""

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,Activation, Flatten, Dense

class ShallowNet:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        inputShape=(height,width,depth)
        model.add(Conv2D(32, (3,3),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model
#%%
if __name__=='__main__':
    s=ShallowNet.build(100,200,3,10)
                  
                 
                  