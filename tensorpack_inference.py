# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:20:15 2019

@author: Sirius
"""

# inference from .meta and checkpoints file

# check all nodes name from following
# output_nodes_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
# Note: inference nodes name different from training

import os
import tensorflow as tf


model_dir = r'E:\denseNet\densenet-tensorflow-master\train_log\mixnet\cifar10_share\k_16_0306-191954'

# one meta file in each saved dierctory
meta_file_name = [s for s in os.listdir(model_dir) if s.endswith('.meta')]
assert len(meta_file_name) == 1 ,'more than one meta file'
meta_path = os.path.join(model_dir,meta_file_name[0])

from predict import cifar10_load_data
data =  cifar10_load_data(r'E:\Data',r'test_batch')
data_tmp = data[0][:50,...]


saver = tf.train.import_meta_graph(meta_path,clear_devices=True)

with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(model_dir))

    #accessing the default graph which we restored
    graph = tf.get_default_graph()
    
    #op that we can be processed to get the output
    #last is the tensor that is the prediction of the network
    x_input = graph.get_tensor_by_name('InferenceTower/sub:0')
    y_pred = graph.get_tensor_by_name('InferenceTower/output:0')
    
    results = sess.run(y_pred, feed_dict = {x_input:data_tmp})

import numpy as np 
res = np.argmax(results, 1)
