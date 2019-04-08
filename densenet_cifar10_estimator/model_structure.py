# 文章中模型的设置，训练细节


# CIFAR10
# ================================================
# densenet:
# 数据预处理:
# normalize data using channel means and std
# mirroring/shifting c10+

# three dense block, each equal number of layers
# first conv filter number: densenet: 16 /densenet-bc: k * 2
# conv: 3x3, zero-padding with feature map size fixed
# transition layer: 1x1 conv followed by 2x2 avg pooling (between two dense block)
# global avg pooling --> softmax
# three blocks: 32x32, 16x16, 8x8
# bottleneck: compress ratio=0.5

# SGD
# batch_size: 64
# epochs: 300
# init learning rate: 0.1, 0.01, 0.001 (150, 225, 300)
# weight decay: 1e-4
# momentum: 0.9
# kernel_initialize: 

# densenet: d_40_k_12, d_100_k_12, d_100_k_24
# densenet-bc: d_100_k_12, d_250_k_24, d_190_k_40


# Note: dataset without data augmentation: dropout after each conv layer with p=0.2 (except first one)
# densenet:    3n+4
# densenet-bc: 6n+4



# ================================================
# ResNext:

# 数据预处理：
# 1.zero-padding(to 40) --> center cropping
# 2.flipping

# init conv: filter num = 64
# bottleneck block: (1x1,64) --> (3x3,64) --> (1x1,256)
# global avg pooling -> fc
# downsampling between two stage: conv with stride 2
# 每个stage的最后一个block用

# training:
# batch_size=128
# weight decay: 0.0005
# momentum: 0.9
# lr: 0.1 
# epoch:300, lr decay at 150 and 225
