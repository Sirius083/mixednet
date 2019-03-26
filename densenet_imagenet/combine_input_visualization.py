#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from densenet_preprocessing import *

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# pretrained model dir
model_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt'
meta_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt.meta'
pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'

# ===============================
def get_batch(image_dir, batch_size):
    all_images = os.listdir(image_dir)
    filelist = [os.path.join(image_dir, path) for path in all_images]

    file_queue = tf.train.string_input_producer(filelist)

    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)

    image = tf.image.decode_jpeg(value, channels = 3)

    image = preprocess_image(image, output_height = 224, output_width = 224, 
                     is_training=False,resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX)

    image_batch = tf.train.batch([image],batch_size=batch_size)

    return image_batch

image_dir = r'E:\ImageNet2012\ILSVRC2012_img_val'
batch_size = 64

# =================================
# load pre-trained graph
graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'
config = tf.ConfigProto(allow_soft_placement = True)
with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name = "")
    # 这里相当于对restore的pb文件新增加了一个node, 输入node
    image_batch = get_batch(image_dir, batch_size)
    print('=============== image_batch', image_batch)

input_name  =  "Placeholder:0"
output_name =  "densenet121/predictions/Softmax:0"
inputs = graph.get_tensor_by_name(input_name)
outputs = graph.get_tensor_by_name(output_name)


layer_name = "densenet121/dense_block3/conv_block1/x2/Conv/convolution:0"
layer = graph.get_tensor_by_name(layer_name)
print('layer ***', layer) 
neuron = layer[:,12,11,0]
print('neuron ***',neuron)

# =============================================
# define guided backpropagation
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y


# get activation
with graph.gradient_override_map({'Relu':'GuidedRelu'}):
     with tf.name_scope('guided_back_pro_map'):
          guided_back_pro = tf.gradients(neuron, inputs)
          print('guided_back_pro',guided_back_pro)


with tf.Session(graph = graph) as sess:
   coord = tf.train.Coordinator()

   threads = tf.train.start_queue_runners(sess,coord=coord)

   for i in range(5):
     image_value = sess.run(image_batch)
     # print('image_value shape', image_value.shape)
     # print('image_preprocessed type', type(image_preprocessed))

     guided_back_pro_value = sess.run(guided_back_pro, feed_dict = {inputs:image_value})
     print('*** guided_back_pro_value', guided_back_pro_value[0].shape)

   coord.request_stop()
   coord.join(threads)






