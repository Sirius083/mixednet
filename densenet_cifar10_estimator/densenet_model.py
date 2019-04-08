# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
replace syntax in sublime: ^.*moving_variance.*n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

# 重复卷积层是否重新定义BN
# bn_share = True

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, name = None):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True, name = name)

################################################################################
# ResNet block definitions.
################################################################################
def _add_layer(self, inputs, block_num, training):
    # block_number: block index in stage
    # c = inputs.get_shape().as_list()[3] # input channel
    name = 'dense_layer.{}'.format(block_num)
    
    with tf.variable_scope(name):
         inputs_new = batch_norm(inputs, training, name = 'bn1')
         inputs_new = tf.nn.relu(inputs_new)
         inputs_new = tf.layers.conv2d(inputs=inputs_new, filters=self.k, kernel_size=3, strides=1,
                                        padding='SAME', use_bias=False, 
                                        kernel_initializer=tf.variance_scaling_initializer(),name = 'conv1')
    inputs = tf.concat([inputs, inputs_new], 3)

    return inputs


def _add_bottleneck_layer(self, inputs, block_num, training):
    # block_number: block index in stage
    # c = inputs.get_shape().as_list()[3] # input channel
    name = 'dense_layer.{}'.format(block_num)
    
    with tf.variable_scope(name):
         inputs_new = batch_norm(inputs, training, name = 'bn1')
         inputs_new = tf.nn.relu(inputs_new)
         inputs_new = tf.layers.conv2d(inputs=inputs_new, filters=self.k, kernel_size=1, strides=1,
                                        padding='SAME', use_bias=False, 
                                        kernel_initializer=tf.variance_scaling_initializer(),name = 'conv1')

         inputs_new = batch_norm(inputs, training, name = 'bn2')
         inputs_new = tf.nn.relu(inputs_new)
         inputs_new = tf.layers.conv2d(inputs=inputs_new, filters=self.k * self.expansion, kernel_size=3, strides=1,
                                        padding='SAME', use_bias=False, 
                                        kernel_initializer=tf.variance_scaling_initializer(),name = 'conv2')

    inputs = tf.concat([inputs, inputs_new], 3)

    return inputs


def _add_transition(self, inputs, name, training):
    in_channel = inputs.get_shape().as_list()[3]

    if self.bottleneck:
       out_channel = max(int(math.floor(in_channel//self.compressionRate)), self.k)
    else:
       out_channel = in_channel

    with tf.variable_scope(name):
         inputs = batch_norm(inputs, training, name = 'bn1')
         inputs = tf.nn.relu(inputs)
         inputs = tf.layers.conv2d(inputs=inputs, filters=out_channel, kernel_size=1, strides=1,
                                   padding='SAME', use_bias=False, 
                                   kernel_initializer=tf.variance_scaling_initializer(),name = 'conv1')
         inputs = tf.layers.average_pooling2d(inputs, pool_size = 2, strides = 2)
    return inputs


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, num_classes, d, k, compressionRate, bottleneck, expansion, dtype=DEFAULT_DTYPE):
    # params:
    # d: model depth
    # k: each layer add k channel
    # compressionRate: transition layer compression Rate
    # num_class: classification class num
    # bottleneck: boolean, use bottleneck structure or not
    # expansion:  bottleneck channel number multiplier

    if dtype not in ALLOWED_TYPES:
       raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.d = d
    self.k = k
    self.compressionRate = compressionRate
    self.num_classes = num_classes
    self.data_format = 'channel_last'

    self.bottleneck = bottleneck
    if self.bottleneck:
       assert (self.d - 4) % 6 == 0, 'bottleneck, depth should be 6n + 4'
       self.N = int((d - 4)  / 6) # block number in each stage
       self.expansion = expansion
    else:
       assert (self.d - 4) % 3 == 0, 'non bottleneck, depth should be 3n + 4'
       self.N = int((d - 4)  / 3)


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)


  def _model_variable_scope(self):
      return tf.variable_scope('densenet_model',
                               custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):

    inputs = tf.identity(inputs, 'model_inputs')
    print('===================== model inputs', inputs)

    with self._model_variable_scope():

      # init conv
      if self.bottleneck:
         init_channel_num = self.k * 2
      else:
         init_channel_num = 16
      inputs = tf.layers.conv2d(inputs=inputs, filters=init_channel_num, kernel_size=3, strides=1,
                                        padding='SAME', use_bias=False, 
                                        kernel_initializer=tf.variance_scaling_initializer(),name = 'init_conv')

      if not self.bottleneck:
         with tf.variable_scope('stage1'):
              for block_num in range(self.N):
                  inputs = _add_layer(self, inputs, block_num, training)
              inputs = _add_transition(self, inputs, 'transition1', training)

         with tf.variable_scope('stage2'):
              for block_num in range(self.N):
                  inputs = _add_layer(self, inputs, block_num, training)
              inputs = _add_transition(self, inputs, 'transition2', training)


         with tf.variable_scope('stage3'):
              for block_num in range(self.N):
                  inputs = _add_layer(self, inputs, block_num, training)

      if self.bottleneck:
         with tf.variable_scope('stage1'):
              for block_num in range(self.N):
                  inputs = _add_bottleneck_layer(self, inputs, block_num, training)
              inputs = _add_transition(self, inputs, 'transition1', training)

         with tf.variable_scope('stage2'):
              for block_num in range(self.N):
                  inputs = _add_bottleneck_layer(self, inputs, block_num, training)
              inputs = _add_transition(self, inputs, 'transition2', training)


         with tf.variable_scope('stage3'):
              for block_num in range(self.N):
                  inputs = _add_bottleneck_layer(self, inputs, block_num, training)


      inputs = batch_norm(inputs, training, name = 'bnlast')
      inputs = tf.nn.relu(inputs)
      
      # global avg pooling
      inputs = tf.reduce_mean(inputs, [1,2], keepdims=True, name = 'final_reduce_mean')
      inputs = tf.squeeze(inputs, [1,2])

      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)

      inputs = tf.identity(inputs, 'final_dense')
      print('===================== model outputs', inputs)

      return inputs

def test_graph():
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    
    import numpy as np
    input_tensor = tf.zeros([64, 32, 32, 3], dtype=tf.float32)

    # bottleneck version
    model = Model(d=40, k=12, compressionRate=2, num_classes=10, bottleneck=False, 
                  expansion = 4, dtype=DEFAULT_DTYPE)
    
    model(input_tensor, True) # define model
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    # print all variables
    print('=============================================================')
    allvar = tf.global_variables()
    for var in allvar:
        # if 'batch' not in var.name:
          print(var)
    print('=============================================================')
    
    # tensorflow total parameter count
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('model total parameter', total_params/1e6)

if __name__ == '__main__':
    test_graph()

