# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:39:01 2019

@author: Sirius

save model to pb file, then evaluating
"""
# Question: all results print 8
# inference from a pb file 
# 'input' refers to an operation, 'input:0' is the tensorname 


import os

# disable GPU use
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 


import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from math import ceil
from predict import cifar10_load_data


# parameters =============================
model_dir = r'E:\denseNet\densenet-tensorflow-master\train_log\mixnet\cifar10_share\k_16_0306-191954'
input_name  = 'InferenceTower/sub:0'
output_name = 'InferenceTower/output:0'    # Output nodes

batch_count =  1000

data = cifar10_load_data(r'E:\Data','test_batch')
print('Image being loaded....')

groups = ceil(len(data[1])/batch_count)


# graph  ====================================
tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement = True)
graph_pb_path = os.path.join(model_dir,'output_graph.pb')


def load_graph(graph_pb_path):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(graph_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


graph = load_graph(graph_pb_path)
# evaluation ============================================
inputs  = graph.get_tensor_by_name(input_name)
outputs = graph.get_tensor_by_name(output_name)

results = []

with tf.Session(graph=graph) as sess:
     print('Starting evaluation ......')
     for i in range(groups):
        # Note: 这里不能写 [i: i*batch_count + i]
        batch_images = data[0][i * batch_count : i * batch_count + batch_count,...]
        res = sess.run(outputs, {inputs:batch_images})
        res = np.argmax(res,1) # find label
        results = results + list(res)
     print('Evaluationn done.......')

error_rate = 1- np.sum(results == data[1])/len(data[1])

print('error_rate ', error_rate)


# Notes =============================
''' 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
prob = softmax(np.squeeze(res))   
'''

# all graph node names
# all_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
