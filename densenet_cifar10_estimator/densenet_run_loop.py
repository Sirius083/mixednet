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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.

Q:
1. tensorboard中的globalstep是什么？
2. learning_rate: 为什么中间epoch重零计数在tensorboard中显示正确？
3. test new view
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

# from official.resnet import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
# pylint: enable=g-bad-import-order

# from official.utils.export import export
import export
import densenet_model

################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)
  
  print('num_gpus',num_gpus)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):  # pylint: disable=unused-argument
    return model_helpers.generate_synthetic_data(
        input_shape=tf.TensorShape([batch_size, height, width, num_channels]),
        input_dtype=tf.float32,
        label_shape=tf.TensorShape([batch_size]),
        label_dtype=tf.int32)

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  # initial_learning_rate = 0.1 * batch_size / batch_denom
  initial_learning_rate = 0.1
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  # sirius: 重启时epoch是重新开始计数的，这里是否按照step记数
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def densenet_model_fn(features, labels, mode, model_class,
                    weight_decay, learning_rate_fn, momentum,
                    loss_scale, loss_filter_fn=None, dtype=densenet_model.DEFAULT_DTYPE):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  # tf.summary.image('images', features, max_outputs=6)
  
  # 这里是来自模型预处理后的输入
  features = tf.cast(features, dtype)
  
  # print('*** model_class', model_class)
  model = model_class()

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.cast(logits, tf.float32)

  
  # 在用estimator进行predict时候用的
  # 
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor') # sirius: 在这一步进行了softmax
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )
    '''
    # 原版代码：   
    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      minimize_op = optimizer.minimize(loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
    '''
    # 将batch_normalization顺序改变
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]

      with tf.control_dependencies(update_ops):
           # print('**************unscaled_grad_vars',unscaled_grad_vars)
           # print('**************global_step',global_step)
           train_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      # ??? !!! 很神奇
      with tf.control_dependencies(update_ops):
           # print('**************optimizer',optimizer)
           # print('**************loss',loss)
           # print('**************global_step',global_step)
           train_op = optimizer.minimize(loss, global_step)
           # print('*****************train_op before', train_op)
           train_op = tf.group(train_op)
           # print('*****************train_op after', train_op)
  else:
    train_op = None
  
  # sirius: 计算不正确的原因
  # tf.metrics.accuracy: 简单的讲，它计算的是整个session生存期内，所有feed_dict中的数据的正确率
  # http://blog.stackoverflow.club/tf.metrics.accuracy-tensorflow/
  # https://github.com/tensorflow/tensorflow/issues/15115
  if not tf.contrib.distribute.has_distribution_strategy(): # 有多台GPU返回True, 否则返回False
    accuracy = tf.metrics.accuracy(labels, predictions['classes']) # shiguang: 分布式训练，多个GPU时运算
  else:
    # Metrics are currently not compatible with distribution strategies during
    # training. This does not affect the overall performance of the model.
    # accuracy = (tf.no_op(), tf.constant(0)) # shiguang: 输出training_accuracy全是零
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])

    # 时光师兄增加
    # 训练时每个batch的精度，而不是从开始到现在的ema
    # tf.squeeze: 删掉维度为1的axis
    my_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.squeeze(labels), tf.int64), predictions['classes']), tf.float32))
    tf.identity(my_acc, name='train_accuracy_perbatch')
    tf.summary.scalar('train_accuracy_perbatch', my_acc)
    
  # shiguang: 这个是从训练开始到现在的精度
  # 之前用的是在整个training data上的精度，但是这里整个training data load不进来；但是用batch的
  # shiguang: 减小 batch_size
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def densenet_main(
    flags_obj, model_function, input_function, dataset_name, shape=None):
  """Shared main loop for ResNet Models.

  Args:
    flags_obj: An object containing parsed flags. See define_densenet_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.export_dir is passed.
  """

  model_helpers.apply_clean(flags.FLAGS)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  '''
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=1,
      intra_op_parallelism_threads=1,
      allow_soft_placement=True)
  '''
  
  session_config = tf.ConfigProto(allow_soft_placement=True)

  # sirius:
  distribution_strategy = distribution_utils.get_distribution_strategy(
      flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy, session_config=session_config,
      save_summary_steps=500)
  
  # print all flags inside model main
  # for k,v in tf.flags.FLAGS.__flags.items():
  # print('=================================')
  # for k,v in flags_obj.items():
  #     print('***',v.__dict__['name'],v.__dict__['_value'])
  
  # Note: 这里的flags_obj定义了多种类型的flags
  # print(flags_obj)

  train_dir = r'E:\denseNet\resnet_cifar10\train_dir'
  export_dir_all = r'E:\denseNet\resnet_cifar10\export_dir'
  model_name = 'd_{}_k_{}'.format(flags.FLAGS.d, flags.FLAGS.k)
  model_dir = os.path.join(train_dir, model_name)
  export_dir = os.path.join(export_dir_all, model_name)

  # Note flags
  # parameters that will be passed into model fn
  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=model_dir, config=run_config,
      params={
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'loss_scale': flags_core.get_loss_scale(flags_obj),
          'dtype': flags_core.get_tf_dtype(flags_obj),
          # network parameters
          'd': flags_obj.d,
          'k':flags_obj.k,
          'compressionRate':flags_obj.compressionRate,
          'expansion':flags_obj.expansion,
          'bottleneck':flags_obj.bottleneck
      })
  
  # Note flags
  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
  }
  
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('densenet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=model_dir,
      batch_size=flags_obj.batch_size)

  def input_fn_train():
    return input_function(
        is_training=True, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=flags_obj.epochs_between_evals,
        num_gpus=flags_core.get_num_gpus(flags_obj))

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1)

  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals)

  # print('*** total_training_cycle',total_training_cycle)

  for cycle_index in range(total_training_cycle):
    tf.logging.info('Starting a training cycle: %d/%d',
                    cycle_index, total_training_cycle)

    classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                     max_steps=flags_obj.max_train_steps)

    tf.logging.info('Starting to evaluate.')

    # flags_obj.max_train_steps is generally associated with testing and
    # profiling. As a result it is frequently called with synthetic data, which
    # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
    # eval (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=flags_obj.max_train_steps)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results['accuracy']):
      break
    
  # export model at last
  input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
       shape, batch_size=flags_obj.batch_size)
  classifier.export_savedmodel(export_dir, input_receiver_fn)


def define_densenet_flags():
  """Add flags and validators for DenseNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  # print('==============================')
  # 这一部分定义的相当于 params

  flags.DEFINE_integer('d', 40, 'Depth of model')
  flags.DEFINE_integer('k', 12, 'Densenet k')
  flags.DEFINE_integer('compressionRate', 2, 'Densenet transition layer compression rate')
  flags.DEFINE_integer('expansion', 4, 'DenseNet bottleneck structure expansion rate')
  flags.DEFINE_boolean('bottleneck', False, 'Densenet whether to use bottleneck structure or nor')
  
  '''
  all_flags = tf.flags.FLAGS.__flags
  for k,v in all_flags.items():
      print('{:25s}   {}'.format(k, v.default))
  '''

