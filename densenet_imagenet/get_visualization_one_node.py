#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from densenet_preprocessing import *
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# pretrained model dir
model_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt'
meta_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt.meta'
pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'

# image path
IM_PATH = r'E:\denseNet\densenet_imagenet\data'
SAVE_DIR = r'E:\denseNet\densenet_imagenet\data\save_dir'


# ===============================
# pre-process test image
image_path = r'E:\denseNet\densenet_imagenet\data\class_test\003.backpack\003_0020.jpg'
with open(image_path, 'rb') as f:
    contents = f.read()

image = tf.image.decode_jpeg(contents, channels = 3)

image_preprocessed = preprocess_image(image, 224, 224, is_training=False,
                                    resize_side_min=_RESIZE_SIDE_MIN,
                                    resize_side_max=_RESIZE_SIDE_MAX)
image_preprocessed = tf.expand_dims(image_preprocessed, axis = 0)
print('image_preprcessed',image_preprocessed)  


with tf.Session() as sess:
    image_preprocessed = sess.run(image_preprocessed)

# =================================
# load graph
graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'

input_name  =  "Placeholder:0"
output_name = "densenet121/predictions/Softmax:0"
tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement = True)

with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")


inputs = graph.get_tensor_by_name(input_name)
outputs = graph.get_tensor_by_name(output_name)
concat1 = graph.get_tensor_by_name('densenet121/transition_block1/AvgPool2D/AvgPool:0')
concat2 = graph.get_tensor_by_name('densenet121/transition_block2/AvgPool2D/AvgPool:0')
concat3 = graph.get_tensor_by_name('densenet121/transition_block3/AvgPool2D/AvgPool:0')

class_act = [concat1, concat2, concat3]

# define guided backpropagation
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y



# get activation
with graph.gradient_override_map({'Relu':'GuidedRelu'}):
     with tf.name_scope('guided_back_pro_map'):
          # guided_back_pro_list = []
          guided_back_pro = tf.gradients(concat3, inputs)
print('guided_back_pro',guided_back_pro)


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(guided_back_pro, {inputs:image_preprocessed})
    # print('res', res[0].shape) # (1,224,224,3)

import scipy
import numpy as np
image_save_path = r'E:\denseNet\densenet_imagenet\data\save_dir\bag.png'
scipy.misc.imsave(image_save_path, np.squeeze(res))





'''
# =================================
# restore pre-trained model: pb, variables(ckpt.data, ckpt.index)
# reload all weights to corresponding variables
from densenet import *
densenet121 = densenet(inputs,num_classes=1000, reduction=0.5,growth_rate=32,num_filters=64,
                       num_layers=[6,12,24,16],data_format='NHWC',is_training=True,reuse=False,
                       scope='densenet121')
'''



