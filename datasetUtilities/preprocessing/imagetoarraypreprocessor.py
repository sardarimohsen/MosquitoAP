#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:25:48 2020

@author: mohsen
"""
from tensorflow.keras.preprocessing.image import img_to_array
class ImageToArrayPreprocessor:
    def __init__(self,dataFormat=None):
        self.dataFormat=dataFormat
        
    def preprocess(self,image):
        return img_to_array(image,data_format=self.dataFormat)
