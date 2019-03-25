#!/usr/bin/env python
# -*- coding: utf-8 -*-
# visualizing one neuron

import numpy as np
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
# image_path = r'E:\denseNet\densenet_imagenet\data\class_test\003.backpack\003_0020.jpg'
image_dir = r'E:\denseNet\densenet_imagenet\visualization\layer_name\pic_val'
import os
filenames = os.listdir(image_dir)
filenames = [os.path.join(image_dir, name) for name in filenames]


def parse_function(filename):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


batch_size = 60
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(lambda image: preprocess_image(image, output_height = 224, output_width = 224, 
                                                     is_training=False,
                                                     resize_side_min=_RESIZE_SIDE_MIN,                                                    resize_side_max=_RESIZE_SIDE_MAX))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
    image_preprocessed = sess.run(next_element) # (60,224,224,3)



# =================================
# load graph
graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'

input_name  = "Placeholder:0"
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

layer_name = 'densenet121/dense_block3/conv_block1/x2/Conv/convolution:0'
layer = graph.get_tensor_by_name(layer_name) # (?,13,13,32)
print('layer ***', layer) 

neuron = layer[:,12,11,0]




# define guided backpropagation
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y

# get activation
with graph.gradient_override_map({'Relu':'GuidedRelu'}):
     with tf.name_scope('guided_back_pro_map'):
          # guided_back_pro = tf.gradients(layer, inputs)
          guided_back_pro = tf.gradients(neuron, inputs)


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run(guided_back_pro, {inputs:image_preprocessed})
    print('*** res.shape', results[0].shape)


# ===============================
# visualizing one neuron
import os
import scipy
import numpy as np
save_img_dir = r'E:\denseNet\densenet_imagenet\visualization\layer_name' 

for i in range(60):
    image_save_path = os.path.join(save_img_dir, str(i)+'.png')
    print('image_save_path',image_save_path)
    print('***', results[0][i].shape)
    scipy.misc.imsave(image_save_path, np.squeeze(results[0][i]))

