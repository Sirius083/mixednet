# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:29:46 2019

@author: Sirius
"""

# test on mixnet pretrained model on cifar10

def cifar10_load_data(data_path, filename):
    # load a pickled data-file from the cifar10
    # params:
    # data_path: directory to store all data
    # filename: 'test_batch','data_batch_1',...

    # data augmentation: 
    # 1. data / 128.0 - 1
    # 2. subtract per pixel mean

    import os
    import numpy as np
    import pickle

    # data_path = r'E:\Data'
    file_path = os.path.join(data_path, 'cifar-10-batches-py', filename)
    print("Loading data: "+ file_path)

    with open(file_path, mode = 'rb') as file_:
        # It is important to set the encoding
        data = pickle.load(file_, encoding = 'bytes')
    
    raw_images = data[b'data']         # (10000,3072)
    labels = np.array(data[b'labels']) # (10000,)

    # images = _convert_images(raw_images)
    # convert the raw images from the data-file to floating-points

    raw_images = np.array(raw_images, dtype = float)

    # reshape the array to 4-dimensions
    images = raw_images.reshape([-1,3, 32, 32])

    # reorder the indicies of the array
    images = images.transpose([0,2,3,1])
    
    
    '''
    # show images
    import scipy.misc
    rgb = scipy.misc.toimage(images[1000])
    rgb
    '''
    
    # 注：原代码中的顺序是
    # 1. images - pp_mean; 2. images/128.0 - 1.0
    # 我的预处理代码是：
    # 1. images/128.0 - 1; 2. images - pp_mean;
    
    
    # data augmentation: subtract per pixel mean in dataset
    pp_mean = np.mean(images, axis=0)
    images = images -  pp_mean
    images = images / 128.0 - 1
    
    return images, labels



'''
data_path = 'E:\Data'
filename = 'test_batch'
test_data = cifar10_load_data(data_path,filename)
'''









