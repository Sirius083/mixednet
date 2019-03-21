#!/usr/bin/env python
# -*- coding: utf-8 -*-

# read imagenet validation jpeg file to batch tensor
import tensorflow as tf
from densenet_preprocessing import *

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# pretrained model dir
model_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt'
meta_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\tf-densenet121.ckpt.meta'
pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'


def parse_function(filename):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(lambda image: preprocess_image(image, output_height = 224, output_width = 224, 
                                                     is_training=False,
                                                     resize_side_min=_RESIZE_SIDE_MIN,
                                                     resize_side_max=_RESIZE_SIDE_MAX))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
# print(dataset)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    # image_preprocessed = sess.run(image_preprocessed)
    value = sess.run(next_element)
    print('value shape',value.shape)
