# https://github.com/joeyearsley/efficient_densenet_tensorflow
# memory efficient implementation of densenet

def _conv_block(self, ip, nb_filter):
    """ Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout

        Args:
            ip: Input tensor
            nb_filter: number of filters

        Returns: tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    """

    def _x(ip):
        x = batch_normalization(ip, **self.bn_kwargs)
        x = tf.nn.relu(x)

        if self.bottleneck:
            inter_channel = nb_filter * 4

            x = conv2d(x, inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       **self.conv_kwargs)
            x = batch_normalization(x, **self.bn_kwargs)
            x = tf.nn.relu(x)

        x = conv2d(x, nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
                   **self.conv_kwargs)

        if self.dropout_rate:
            x = dropout(x, self.dropout_rate, training=self.training)

        return x

    if self.efficient:
        # efficient implementation: recompute gradient on the backward pass
        # Gradient checkpoint the layer
        _x = tf.contrib.layers.recompute_grad(_x)

    return _x(ip)
