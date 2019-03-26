#!/usr/bin/env python
# -*- coding: utf-8 -*-

# add dataset as input to the pb graph

import tensorflow as tf
from densenet_preprocessing import *

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# pretrained model dir
model_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt'
meta_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt.meta'
pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'

# ===============================
import os
image_dir = r'E:\ImageNet2012\ILSVRC2012_img_val'
all_images = os.listdir(image_dir)
filenames = [os.path.join(image_dir, path) for path in all_images]

def parse_function(filename):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


input_name  =  "Placeholder:0"
output_name =  "densenet121/predictions/Softmax:0"

# batch_tmp = tf.placeholder(dtype = tf.float32, shape = (None, 224,224,3), name = "batch_tmp")
# print('batch_tmp', batch_tmp)


graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'
config = tf.ConfigProto(allow_soft_placement = True)
with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # print('graph_def', type(graph_def))
    # allname = [n.name for n in graph_def.node]

'''
print('=================')
for name in allname:
    print(name)
'''

tf.reset_default_graph()
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(lambda image: preprocess_image(image, output_height = 224, output_width = 224, 
                                                     is_training=False,
                                                     resize_side_min=_RESIZE_SIDE_MIN,                                                    resize_side_max=_RESIZE_SIDE_MAX))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


[y] = tf.import_graph_def(graph_def, input_map = {"Placeholder:0":next_element}, 
                    return_elements=["densenet121/predictions/Softmax:0"], name = "")

print('y', y)


'''
tf.import_graph_def(graph_def, input_map = {"Placeholder:0":next_element}, name = "")
graph = tf.get_default_graph()
print('graph', graph)
'''
