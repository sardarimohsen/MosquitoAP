#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:37:48 2020

@author: mohsen
"""

if __name__=='__main__' :
    from shallownet import ShallowNet
    from lenet import LeNet
    print('INFO:nn init')
else:
    from .shallownet import ShallowNet
    from .lenet import LeNet
    print('INFO:nn init')
