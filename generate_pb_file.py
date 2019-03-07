# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:38:00 2019

@author: Sirius
"""
'''
Do not use a metagraph for inference
It is common to change the graph for inference
Different data layout for CPU infenrence
'''
# tensorflow export saved ckpt file to pb file for efficient inference
# A blog on pb graph inference
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

# 只load一个output的节点不报错，但是预测分类结果全是8
# load 所有nodes, 报错

import tensorflow as tf
import os

def generate_pb_file(model_dir,output_node_name):
    '''
    parameters: 
    model_dir: the output model directory
    input_node_name: input name in graph
    output_node_name: output node name in graph
    all_name = [n.name for n in tf.get_default_graph().as_graph_def().node]

    Set the name when defining the model
    '''    
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement = True)
    # one meta file in each saved dierctory
    allfiles = os.listdir(model_dir)
    pb_file_name = [s for s in allfiles if s.endswith('.meta')]
    assert len(pb_file_name) == 1 ,'more than one meta file'
    pb_file_name = pb_file_name[0]
    meta_path = os.path.join(model_dir,pb_file_name)

    with tf.Session(config = config) as sess:

        # Restore the graph
        # clear_divices: do not care which GPU to use
        saver = tf.train.import_meta_graph(meta_path,clear_devices=True)

        # Load weights
        saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        
        # 不能只export一个node, 要export所有node
        # output_nodes_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_name)
        
        # generate corresponding file in the model checkpoint directory
        graph_pb_path = os.path.join(model_dir,'output_graph.pb')
        # Save the frozen graph
        with open(graph_pb_path, 'wb') as f:
             f.write(frozen_graph_def.SerializeToString())

    print('Save model pb file to path, ', graph_pb_path)

model_dir = r'E:\denseNet\densenet-tensorflow-master\train_log\mixnet\cifar10_share\k_16_0306-191954'
# input_node_name = 'input'
output_node_name = ['InferenceTower/output']    
generate_pb_file(model_dir,output_node_name)
