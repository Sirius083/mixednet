import tensorflow as tf

'''
# restore graph (built from scratch here for the example)
x = tf.placeholder(tf.int64, shape=(), name='x')
y = tf.square(x, name='y')
print('*** x', x)
print('*** y', y)

# just for display -- you don't need to create a Session for serialization
with tf.Session() as sess:
  print("with placeholder:")
  for i in range(10):
    print(sess.run(y, {x: i}))

# serialize the graph
graph_def = tf.get_default_graph().as_graph_def()
print('graph_def', type(graph_def))

tf.reset_default_graph()
# print all graph node --> after reset, no graph node

# build new pipeline
batch = tf.data.Dataset.range(10).make_one_shot_iterator().get_next()
print('*** batch', batch)

# plug in new pipeline
[y] = tf.import_graph_def(graph_def, input_map={'x:0': batch}, return_elements=['y:0'])

# enjoy Dataset inputs!
with tf.Session() as sess:
  print('with Dataset:')
  try:
    while True:
      print(sess.run(y))
  except tf.errors.OutOfRangeError:
    pass        
'''


# tensorflow 批量读取图片数据
# https://blog.csdn.net/XUEER88888888888888/article/details/86666614
# tensorflow read multiple pictures at one time
import tensorflow as tf
import os

from densenet_preprocessing import *

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

#批处理大小，跟队列，数据的数量没有影响，只决定 这批次处理多少数据

    
# 1.找到文件，放入列表  路径+名字  ->列表当中
# image_dir = r'E:\ImageNet2012\ILSVRC2012_img_val'
image_dir = r'E:\denseNet\densenet_imagenet\visualization\layer_name\pic_val'
all_images = os.listdir(image_dir)
filelist = [os.path.join(image_dir, path) for path in all_images]
# image_batch= picread(filelist)

# 1.构造文件的队列
file_queue = tf.train.string_input_producer(filelist)
print('file_queue',file_queue)

# 2.构造阅读器去读取图片内容（默认读取一张图片）
reader = tf.WholeFileReader()
key,value = reader.read(file_queue)

# 3.对读取的图片进行解码
image = tf.image.decode_jpeg(value, channels = 3)
print('image ***', image) # (?,?,?)

image = preprocess_image(image, output_height = 224, output_width = 224, 
                 is_training=False,resize_side_min=_RESIZE_SIDE_MIN,
                 resize_side_max=_RESIZE_SIDE_MAX)
print('image after preprocessed ***', image) # (224,224,3)

# 5.进行批处理
image_resize_batch = tf.train.batch([image],batch_size=64)
print('image_resize_batch',image_resize_batch)


#开启会话运行结果
with tf.Session() as sess:
   #定义一个线程协调器
   coord = tf.train.Coordinator()

   #开启读文件的线程
   threads = tf.train.start_queue_runners(sess,coord=coord)

   #打印读取的内容
   for i in range(10):

       value = sess.run(image_resize_batch)
       print('*** i', i, '****** image_batch shape', value.shape)
       # print(value[0,0,:])


   #回收子线程
   coord.request_stop()
   coord.join(threads)

