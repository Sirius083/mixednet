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
# sirius: 处理的是数据版本是cifar10-bin文件(binary文件)
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('E:\\resnet\\models-master') # sirius

import os
# tensorflow disable GPU use
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger

import densenet_model
import densenet_run_loop

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS

# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'

# import numpy as np
# cifar10_mean = np.array((0.4914, 0.4822, 0.4465))
# cifar10_std = np.array((0.2470, 0.2435, 0.2616))
# cifar100_mean = np.array((0.5071, 0.4865, 0.4409))
# cifar100_std = np.array((0.2673, 0.2564, 0.2762))



###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  # data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
  data_dir = r'E:\Data\cifar-10-batches-bin' # sirius
  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  # sirius: Reinterpret the bytes of a string as a vector of numbers.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES], [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
  
  # print('*** preprocess_image', image)
  # Subtract off the mean and divide by the variance of the pixels.
  # image = tf.image.per_image_standardization(image)
  
  # constant must from the same file
  CIFAR10_MEAN = tf.constant((0.4914, 0.4822, 0.4465))
  CIFAR10_STD = tf.constant((0.2470, 0.2435, 0.2616))
  image = (image - CIFAR10_MEAN)/CIFAR10_STD
  # print('*** preprocess_image', image)

  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES) # sirius: 这里输入的文件名可以是一个list

  return densenet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train'],
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(densenet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, num_classes=_NUM_CLASSES):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    #  data_format = 'channels_last', dtype = tf.float32

    # need model positional argument
    # 'd', 'k', 'compressionRate', 'bottleneck', and 'expansion'
    # print current param values
    '''
    for k,v in tf.flags.FLAGS.__flags.items():
        # print('***',v.__dict__['name'],v.__dict__['_value'])
        print(k,v)
    '''
    # print('=======================')
    # print(tf.flags.FLAGS.__flags)

    # 这里定义模型框架的具体参数
    # print('***&&& Inside Cifar10Model', tf.flags.FLAGS.__flags)
    # get current all flags
    # print('***&&& flags', flags.FLAGS.compressionRate)
    super(Cifar10Model, self).__init__(num_classes=num_classes, 
                                       d = flags.FLAGS.d,
                                       k = flags.FLAGS.k,
                                       compressionRate = flags.FLAGS.compressionRate, 
                                       bottleneck = flags.FLAGS.bottleneck, 
                                       expansion = flags.FLAGS.expansion)

def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  # print('===============================================')
  # print('inside cifar10_model_fn: params', params)
  
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = densenet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=64,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[150, 225],
      decay_rates=[0.1, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  
  weight_decay = 1e-4 # sirius: 1e-4 original paper 在imagenet数据集上用的是1e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  # including all variables to calculate weight decay
  def loss_filter_fn(_):
    return True

  return densenet_run_loop.densenet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Cifar10Model,
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype'])


def define_cifar_flags():
  densenet_run_loop.define_densenet_flags()
  '''
  print('=================================')
  print('flags', flags.__dict__)
  '''

  flags.adopt_module_key_flags(densenet_run_loop)
  '''
  print('=================================')
  print('flags', flags.__dict__)
  '''
  # define model flags here
  # 在命令行中修改的参数  

  flags_core.set_defaults(data_dir='E:/Data/cifar-10-batches-bin',
                          model_dir =  r'E:\denseNet\resnet_cifar10\train_dir',
                          export_dir =  r'E:\denseNet\resnet_cifar10\export_dir',
                          train_epochs = 300,
                          epochs_between_evals=1,
                          batch_size=64)
  '''
  print('=================================')
  print('flags core', flags_core.__dict__)
  '''

def run_cifar(flags_obj):
  """Run ResNet CIFAR-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                    or input_fn)
  
  '''
  # cifar10: print all flags
  print('=============')
  for k,v in tf.flags.FLAGS.__flags.items():
      print('***',v.__dict__['name'],v.__dict__['_value'])
  '''
  
  densenet_run_loop.densenet_main(
      flags_obj, cifar10_model_fn, input_function, DATASET_NAME,
      shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])
  

def main(_):
  with logger.benchmark_context(flags.FLAGS):
    # Note: flags_obj 是在这里定义的
    # print('flags.FLAGS',flags.FLAGS)
    run_cifar(flags.FLAGS)

if __name__ == '__main__':
  # TensorFlow will tell you all messages that have the label INFO
  tf.logging.set_verbosity(tf.logging.INFO)
  define_cifar_flags()
  # python decorator: need to use callable functions to parse parameters
  absl_app.run(main)
