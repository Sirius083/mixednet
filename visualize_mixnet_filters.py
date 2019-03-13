# coding: utf-8


import os
import sys
import time
import copy
import numpy as np

from tf_cnnvis import *

import tensorflow as tf
from scipy.misc import imread, imresize
from load_cifar_data import cifar10_load_data

# generate pb file from pre-trained checkpoint
graph_pb_path =  r'E:\denseNet\densenet-tensorflow-master\train_log\mixnet_concat\cifar10\d_40_k_12_gap_2\output_min.pb'


# We load the protobuf file from the disk and parse it to retrieve the 
# unserialized graph_def
with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def, name="")


default_graph = tf.get_default_graph()

# get pictures
images, labels = cifar10_load_data('E:/Data', 'test')
images_np = np.expand_dims(images[0,...], axis=0) # 加上第一个维度
labels_np = np.expand_dims(labels[0],axis=0)


# deconv visualization ========================
# 要可视化模型的相关节点

input_name = 'input:0'
input_tensor = default_graph.get_tensor_by_name(input_name)
# output_name = 'InferenceTower/output:0'

# 可视化的是 operation 的名字
# all_node_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
layers = ["InferenceTower/block1/transition1/pool/output",\
          "InferenceTower/block2/transition2/pool/output",\
          "InferenceTower/linear/output"]

# start session
start = time.time()

is_success = deconv_visualization(sess_graph_path = default_graph , value_feed_dict = {input_tensor: images_np}, 
                                  input_tensor=input_tensor, layers=layers, 
                                  path_logdir=os.path.join("Log","MixnetShareExample"), 
                                  path_outdir=os.path.join("Output","MixnetShareExample"))
start = time.time() - start
print("Total Time = %f" % (start))


# 结果分别存在




