#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 计算给定神经元的感受野，在输入图像上
# https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
# 
#Each layer i requires the following parameters to be fully represented: 
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

# Sirius
# The receptive field is defined as the region in the input space that a particular CNN's feature
# is looking at(i.e. be affected by)
# RF can be decribed by its center location and its size
# Within a receptive field, the closer a pixel to the center of the field
# the more it contributes to the calculation of the output feature

import math
# convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
# layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
# imsize = 227

# densenet-121 convolutional structure
# same padding: ceil(k/2)
# 3x3 max pooling: [3,2,1]
# 2x2 average pooling: [2,2,0]
# 卷积层按如下方式定义
# [filter size, stride, padding]
convnet =   [[7,2,3],[3,2,1]] + [[1,1,0],[3,1,1]]*6 + [[1,1,0],[2,2,0]] + [[1,1,0],[3,1,1]]*12+\
            [[1,1,0],[2,2,0]] + [[1,1,0],[3,1,1]]*24 + [[1,1,0],[2,2,0]] + [[1,1,0],[3,1,1]]*16

block1_name = [['block1_conv' + str(i) + '_k1',  'block1_conv' + str(i) + '_k3'] for i in range(1,7)]
block2_name = [['block2_conv' + str(i) + '_k1',  'block2_conv' + str(i) + '_k3'] for i in range(1,13)]
block3_name = [['block3_conv' + str(i) + '_k1',  'block3_conv' + str(i) + '_k3'] for i in range(1,25)]
block4_name = [['block4_conv' + str(i) + '_k1',  'block4_conv' + str(i) + '_k3'] for i in range(1,17)]

# list of list to list
block1_name = [item for sublist in block1_name for item in sublist]
block2_name = [item for sublist in block2_name for item in sublist]
block3_name = [item for sublist in block3_name for item in sublist]
block4_name = [item for sublist in block4_name for item in sublist]

layer_names = ['conv0','maxpool0'] + block1_name + ['trans1_conv','trans1_pool'] +\
                                     block2_name + ['trans2_conv','trans2_pool'] +\
                                     block3_name + ['trans3_conv','trans3_pool'] +\
                                     block4_name
print('layer_names', layer_names)

imsize = 224


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
  
    n_out = math.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out-1)*s - n_in + k 
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)
  
    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out
  
def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
 
layerInfos = []

if __name__ == '__main__':
   #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
   print ("-------Net summary------")
   currentLayer = [imsize, 1, 1, 0.5]
   printLayer(currentLayer, "input image")
   for i in range(len(convnet)):
       currentLayer = outFromIn(convnet[i], currentLayer)
       layerInfos.append(currentLayer)
       printLayer(currentLayer, layer_names[i])
   
   print ("------------------------")
   

   
   # layer_name = raw_input ("Layer name where the feature in: ")
   layer_name = "block3_conv1_k3"
   print('Layer name where the feature in: ', layer_name)
   layer_idx = layer_names.index(layer_name)

   # idx_x = int(raw_input ("index of the feature in x dimension (from 0)"))
   # idx_y = int(raw_input ("index of the feature in y dimension (from 0)"))
   idx_x = 12
   print("index of the feature in x dimension (from 0), idx_x ", idx_x)
   idx_y = 11
   print("index of the feature in x dimension (from 0), idx_y ", idx_y)

   n = layerInfos[layer_idx][0]
   j = layerInfos[layer_idx][1]
   r = layerInfos[layer_idx][2]
   start = layerInfos[layer_idx][3]

   print('n', n)
   print('i', i)
   print('r', i)
   print('start', start)

   assert(idx_x < n)
   assert(idx_y < n)
  
   print ("receptive field: (%s, %s)" % (r, r))
   print ("center: (%s, %s)" % (start+idx_x*j, start+idx_y*j))


'''
pic_path = r'E:\denseNet\densenet_imagenet\visualization\layer_name\tmp\00011614.png'
import cv2
im = cv2.imread(pic_path)
print('im.shape', im.shape)

import numpy as np
nonzero = np.nonzero(im) 
'''
   
