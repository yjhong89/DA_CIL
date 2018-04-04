import tensorflow as tf
import numpy as np

# Instance normalization layer is applied at test time
def _instance_norm(x, training=True, name='instance_norm', epsilon=1e-5):
    _, height, width, channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        # scaling and shift variables in R^channel
        scale = tf.get_variable('scale', [channel], initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift', [channel], initializer=tf.constant_initializer(0))
        # instance specific normalization, do not average at batch axis
            # keep_dims=True for calculate with 'x'
        instance_mean, instance_var = tf.nn.moments(x, [1,2], keep_dims=True)
        
        normalized = (x - instance_mean) / tf.sqrt(instance_var + epsilon)
        
    return normalized

def _batch_norm(x, training=True, name='batch_norm', decay=0.99, epsilon=1e-5):
    _, _, _, channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [channel], initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift', [channel], initializer=tf.constant_initializer(0))

        pop_mean = tf.get_variable('pop_mean', [channel], initializer=tf.constant_initializer(0), trainable=False)
        pop_var = tf.get_variable('pop_var', [channel], initializer=tf.constant_initializer(1), trainable=False)

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(pop_var, pop_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset=shift, scale=scale, variance_epsilon=epsilon)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset=shift, scale=scale, variance_epsilon=epsilon)


def _leaky_relu(x, slope=0.2):
    return tf.maximum(x, 0.2*x)

def residual(x, filter_size, stride, channel, layer_index, name='residual'):
    with tf.name_scope(name):
        residual = tf.identity(x)
        x = conv2d(x, out_channel=channel, filter_size=filter_size, stride=stride, name='conv2d_%d'%layer_index)
        index = layer_index + 1
        x = conv2d(x, out_channel=channel, filter_size=filter_size, stride=stride, name='conv2d_%d'%index)
        index += 1
        return x + residual, index

def fc(x, hidden, dropout_ratio=0.5, activation=tf.nn.relu, dropout=True, name='fc'):
    _, in_dim = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[in_dim, hidden], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[hidden], initializer=tf.constant_initializer(0))
        output = tf.matmul(x, weight)
        output = output + bias

        if dropout:
            output = tf.nn.dropout(output, dropout_ratio, name='dropout')
        if activation:
            output = activation(output)

    return output

def global_average_pooling(x, name='global_pooling'):
    # Make fully connected layer
    return tf.reduce_mean(x, [1,2])
    
def dilated_conv2d(x, out_channel, filter_size, dilation_rate, name='dilated_conv2d', activation=_leaky_relu, normalization=_instance_norm, padding='VALID', training=True):
    _, _, _, in_channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, in_channel, out_channel], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0))
                
        rate = [dilation_rate, dilation_rate]
        # stride must be 1
        output = tf.nn.convolution(x, weight, padding=padding, dilation_rate=rate) + bias

        if normalization:
            output = normalization(output)
        if activation:
            output = activation(output)

        #print('dilated convolution', output.get_shape().as_list())

    return output      


def conv2d(x, out_channel, filter_size=4, stride=2, name='conv2d', activation=_leaky_relu, normalization=_instance_norm, padding='SAME', training=True):
    _, _, _, in_channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, in_channel, out_channel], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0))
        # NHWC: batch, height, width, channel
        # padding VALID: no zero padding, only covers the valid input
        output = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding=padding, data_format='NHWC', name='convolution')
        output = output + bias

        if normalization:
            output = normalization(output, training=training)
        if activation:
            output = activation(output)

        #print('convolution', output.get_shape().as_list())

    return output

# size_out = stride*(size_in) + filter_size - 2*pad
# Literally transposed version of convolution
def transpose_conv2d(x, out_channel, filter_size=4, stride=2, name='transpose_conv2d', activation=tf.nn.relu, normalization=True):
    batch_size, height, width, in_channel = x.get_shape().as_list()
    new_height, new_width = int(height*stride), int(width*stride)
    out_shape = tf.stack([batch_size, new_height, new_width, out_channel])
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [filter_size, filter_size, out_channel, in_channel], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [out_channel], initializer=tf.constant_initializer(0))

        output = tf.nn.conv2d_transpose(x, weight, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME', data_format='NHWC', name='transposed_convolution')
        output = output + bias

        if normalization:
            output = _instance_norm(output)
        if activation:
            output = activation(output)
        #print('upsampling', output.get_shape().as_list())
            
        return output        

# To downsize width and height when downsampling
def _max_pool(x, kernel=[1,2,2,1], stride=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding='SAME')
        

def residual_block(x, out_dim, layer_index, dilation_rate=1, filter_size=3, downsample=True, name='residual', normalization=_instance_norm, training=True):
    in_dim = x.get_shape().as_list()[-1]
    if in_dim * 2 == out_dim:
        increase_dim = True
    else:
        increase_dim = False

    with tf.variable_scope(name):
        # int: floor
        pad = int((filter_size - 1) / 2 * dilation_rate)
        r = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], 'SYMMETRIC')
        # Note that when using dilated convolution, stride must be 1
        # When down sampling, [height, width, channel] -> [height/2, width/2, channel*2], so conv2 and x does not match
        # From Resnet, we add zero pads to increase the depth
        if downsample:
            x = _max_pool(x)
            r = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], 'SYMMETRIC')
            r1 = dilated_conv2d(r, out_channel=out_dim, filter_size=filter_size, activation=tf.nn.relu, padding='VALID', dilation_rate=dilation_rate, name='conv2d_%d'%layer_index, normalization=normalization, training=training)
            x = tf.pad(x, [[0,0],[0,0],[0,0],[in_dim//2,in_dim//2]], 'CONSTANT')
            #print(r1.get_shape().as_list())
        else:
            if increase_dim:
                r1 = dilated_conv2d(x, out_channel=out_dim, filter_size=filter_size, activation=tf.nn.relu, padding='SAME', dilation_rate=dilation_rate//2, name='conv2d_%d'%layer_index, normalization=normalization, training=training)
                x = tf.pad(x, [[0,0],[0,0],[0,0],[in_dim//2,in_dim//2]], 'CONSTANT')
            else:
                r1 = dilated_conv2d(r, out_channel=out_dim, filter_size=filter_size, activation=tf.nn.relu, padding='VALID', dilation_rate=dilation_rate, name='conv2d_%d'%layer_index, normalization=normalization, training=training)
        index = layer_index + 1
        r1 = tf.pad(r1, [[0,0],[pad,pad],[pad,pad],[0,0]], 'SYMMETRIC')
        r2 = dilated_conv2d(r1, out_channel=out_dim, filter_size=filter_size, activation=None, padding='VALID', dilation_rate=dilation_rate, name='conv2d_%d'%index, normalization=normalization, training=training)
        #print('layer: %d, shape of x: %s, shape of r: %s' % (layer_index, x.get_shape().as_list(), r2.get_shape().as_list()))
        index += 1
        return tf.nn.relu(r2+x), index



