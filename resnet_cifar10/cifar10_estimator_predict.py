# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 08:45:47 2018

@author: Sirius

sess.close()
tf.reset_default_graph()

tensorflow 官网预训练好的 cifar10 模型(设定了export_dir)
在test_batch上进行预测
"""
inputs_len = 10000
batch_size = 128
# 参考链接 tensorflow estimator pre-trained model for prediction
# https://stackoverflow.com/questions/46098863/how-to-import-an-saved-tensorflow-model-train-using-tf-estimator-and-predict-on

import tensorflow as tf
# resnet cifar10 32layer
# 训练代码中的函数
def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32,
                                           batch_size=1):
  def serving_input_receiver_fn():
    # Prep a placeholder where the input example will be fed in
    features = tf.placeholder(
        dtype=dtype, shape=[batch_size] + shape, name='input_tensor')

    return tf.estimator.export.TensorServingInputReceiver(
        features=features, receiver_tensors=features)

  return serving_input_receiver_fn

export_dir = r'E:\resnet_cifar10\log\resnet_original\export_dir\resnet_20\1555449367'

# tensorflow official resnet pre-processing cifar10 dataset
import numpy as np
import pickle

# cifar10 test batch 数据的读取
test_path = 'E:/resnet_cifar10/cifar10_data/cifar-10-batches-py/test_batch'    
with open(test_path, 'rb') as f:
     datadict = pickle.load(f,encoding='latin1')
     
X = datadict["data"] 
Y = datadict['labels']
X = X.reshape(inputs_len, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
Y = np.array(Y)

# 数据预处理
inputs = tf.constant(X)
label = tf.constant(Y)
# image = tf.image.per_image_standardization(image)
# 对一个batch的数据进行预处理
inputs = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)
input_array = sess.run(inputs)

# 在test batch上进行预测
from tensorflow.contrib import predictor
predict_fn = predictor.from_saved_model(export_dir)


# predictions = predict_fn({"input": input_array[:128,...]})

predictions = []
from math import floor
iter_num = floor(inputs_len / batch_size)
for batch in range(iter_num):
    # print('batch', batch)
    inputs = input_array[batch_size * batch:batch_size * (batch + 1),...]
    pred = predict_fn({"input": inputs})
    predictions = predictions + list(pred['classes'])

# remaining inputs(can not be include into batch)
inputs = input_array[-batch_size:,...]
pred_last = predict_fn({"input": inputs})
pred_last = list(pred_last['classes'])
    
ind = inputs_len - len(predictions)
print('ind',ind)
pred_labels = predictions + pred_last[-ind:]
true_labels = sess.run(label)


match_list = [pred_labels[i] == true_labels[i] for i in range(len(true_labels))]
print('error', 1- np.sum(match_list)/len(pred_labels))



