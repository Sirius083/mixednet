# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:51:37 2019

@author: Sirius
"""

# read output json file to status
import os
import json
from pprint import pprint

# 将每个epoch的验证结果写入json文件
# 就不用从events文件里读取summary数据
model_dir = r'E:\denseNet\densenet-tensorflow-master\train_log\mixnet\cifar10_share\k_16_0306-191954'
json_path = os.path.join(model_dir, 'stats.json')
 
with open(json_path) as f:
    data = json.load(f)

# pprint(data)
val_error = [i['validation_error'] for i in data]


import matplotlib.pyplot as plt
plt.plot(val_error)