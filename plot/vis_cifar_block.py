#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 只能画没有bottleneck结构的heatmap图

# cifar100
# d_40_k_12_no_bottleneck
# 但是conv以后还有 BN--> relu, 这个怎样考虑

# normalized by the input of feature maps

import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorpack import * 
import math

from matplotlib import pyplot as plt
from matplotlib import cm


# cifar100 d_40_k_12_no_bottleneck
model_dir = r'E:\denseNet\mixnet_concat\log\sparse_5\cifar100\fetch=dense_d=40_k=12_btn=False_trancon=False_stc=False'

'''
output_node_name = 'output' 
# generate pb graph and get all variable names
def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

g = tf.Graph()
sess = tf.Session(graph=g)

from sparsenet_5 import Model
params = model_dir.split('\\')[-1]
params = dict(item.split('=') for item in params.split("_"))
model=Model(dataset = 'cifar100', fetch = params['fetch'], shortcut = params['stc'] == 'True', depth = int(params['d']), k = int(params['k']), 
            bottleneck = params['btn'] == 'True', tran_concat = params['trancon'] == 'True', l2_wd = 1e-4, mom=0.9)
with g.as_default():
     input_tensor = [tf.constant(np.zeros([1, 32, 32, 3]), dtype=tf.float32), tf.constant(np.zeros([1]), dtype=tf.int32)]
     with TowerContext('', is_training=False):
          model._build_graph(input_tensor)
     sess.run(tf.global_variables_initializer())
     
     # all parameters
     print('==============================================')
     allvars = tf.trainable_variables()
     for var in allvars:
         if 'bn' not in var.name:
             print(var.name, var.shape)
     print('==============================================')
     
     params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), [output_node_name])
output_graph_path = os.path.join(model_dir, 'graph.pb')
with tf.gfile.GFile(output_graph_path, "wb") as f:
     f.write(output_graph_def.SerializeToString())
'''

# weight_name = 'stage_1/dense_layer.0/inner/conv/W:0'
def get_graph(graph_pb_path):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.gfile.GFile(graph_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph_pb_path = os.path.join(model_dir, 'graph.pb')
graph = get_graph(graph_pb_path)
sess = tf.Session(graph=graph)


# stage 1
corr1 = np.zeros((13,13))
mask = np.tri(corr1.shape[0], k=-1)
corr1 = np.ma.array(corr1, mask=mask)

for i in range(13):
   print('======================================================')
   if i!=12:
      w_name = 'stage_1/dense_layer.' + str(i) + '/inner/conv/W:0'
   else:
      w_name = 'stage_1/transition1/conv1/W:0'

   w_tensor = graph.get_tensor_by_name(w_name)
   w_value = np.mean(np.absolute(sess.run(w_tensor)), axis = (0,1,3))
   c = int((w_value.shape[0]-24)/12)

   if c == 0:
      corr1[0,0] = 1
   else:
      w_split = [24] + [24 + 12 *(i+1) for i in range(c-1)]
      res = np.array_split(w_value, w_split, axis = 0)
      res_sum = [np.sum(d)/len(d) for d in res]
      res_sum = (res_sum - min(res_sum))/(max(res_sum) - min(res_sum))
      corr1[:len(res_sum),i] = res_sum

      print('c', c)
      print('w_split', w_split)
      print('res_sum',res_sum)
       

# stage 2
corr2 = np.zeros((13,13))
mask = np.tri(corr2.shape[0], k=-1)
corr2 = np.ma.array(corr2, mask=mask)

for i in range(13):
   print('======================================================')
   if i!=12:
      w_name = 'stage_2/dense_layer.' + str(i) + '/inner/conv/W:0'
   else:
      w_name = 'stage_2/transition2/conv1/W:0'

   w_tensor = graph.get_tensor_by_name(w_name)
   w_value = np.mean(np.absolute(sess.run(w_tensor)), axis = (0,1,3))
   c = int((w_value.shape[0]-168)/12)

   if c == 0:
      corr2[0,0] = 1
   else:
      w_split = [168] + [168 + 12 *(i+1) for i in range(c-1)]
      res = np.array_split(w_value, w_split, axis = 0)
      res_sum = [np.sum(d)/len(d) for d in res]
      res_sum = (res_sum - min(res_sum))/(max(res_sum) - min(res_sum))
      corr2[:len(res_sum),i] = res_sum

      print('c', c)
      print('w_split', w_split)
      print('res_sum',res_sum)


# stage 3
corr3 = np.zeros((13,13))
mask = np.tri(corr3.shape[0], k=-1)
corr3 = np.ma.array(corr3, mask=mask)

for i in range(13):
   print('======================================================')
   if i != 12:
      w_name = 'stage_3/dense_layer.' + str(i) + '/inner/conv/W:0'
   else:
      w_name = 'linear/W:0'


   w_tensor = graph.get_tensor_by_name(w_name)
   
   if i != 12:
      w_value = np.mean(np.absolute(sess.run(w_tensor)), axis = (0,1,3))
   else:
      w_value = np.mean(np.absolute(sess.run(w_tensor)), axis = (1)) # [456,100]
   
   c = int((w_value.shape[0]-312)/12)

   if c == 0:
      corr3[0,0] = 1
   else:
      w_split = [312] + [312 + 12 *(i+1) for i in range(c-1)]
      res = np.array_split(w_value, w_split, axis = 0)
      # res_sum = [np.sum(d)/len(d) for d in res]
      # scale = len(d)/w_value.shape[0]
      res_sum = [np.sum(d) * len(d)/w_value.shape[0] for d in res]
      res_sum = (res_sum - min(res_sum))/(max(res_sum) - min(res_sum))
      corr3[:len(res_sum),i] = res_sum

      print('c', c)
      print('w_split', w_split)
      print('res_sum',res_sum)
       

# add one color bar to whole picture
corr = [corr1, corr2, corr3]

from matplotlib import pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=3)
cmap = cm.get_cmap('jet') # jet doesn't have white color
cmap.set_bad('w') # default value is 'k'

for i, ax in enumerate(axes.flat):
    im = ax.imshow(corr[i],cmap=cmap)
    ax.title.set_text('Stage_' + str(i+1))

# fig.colorbar(im, ax=axes.ravel().tolist())
# fig.colorbar(im)
plt.show()


'''
from matplotlib import pyplot as plt
fig = plt.figure()
cmap = cm.get_cmap('jet') # jet doesn't have white color
cmap.set_bad('w') # default value is 'k'

ax1 = fig.add_subplot(131)
ax1.imshow(corr1, cmap=cmap)
ax1.title.set_text('Stage_1')

ax2 = fig.add_subplot(132)
ax2.imshow(corr2, cmap=cmap)
ax2.title.set_text('Stage_2')

ax3 = fig.add_subplot(133)
ax3.imshow(corr3, cmap=cmap)
ax3.title.set_text('Stage_3')

fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
'''
