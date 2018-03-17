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

def conv2d(x, out_channel, filter_size=4, stride=2, name='conv2d', activation=_leaky_relu, normalization=True, padding='SAME', dilations=[1,1,1,1]):
    _, _, _, in_channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, in_channel, out_channel], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0))
        # NHWC: batch, height, width, channel
        # padding VALID: no zero padding, only covers the valid input
        output = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding=padding, data_format='NHWC', dilations=dilations, name='convolution')
        output = output + bias

        if normalization:
            output = _instance_norm(output)
        if activation:
            output = activation(output)

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
            
        return output        
        

def residual_block(x, out_dim, layer_index, dilation_rate=1, filter_size=3, downsample=True, name='residual'):
    dilations = [1, dilation_rate, dilation_rate, 1]
    with tf.variable_scope(name):
        # int: floor
        pad = int((filter_size - 1) / 2 * dilation_rate)
        r = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], 'SYMMETRIC')
        # Note that when using dilated convolution, stride must be 1
        if downsample:
            r = op.conv2d(r, out_channel=out_dim, filter_size=filter_size, stride=2, activation=tf.nn.relu, padding='VALID', dilations=dilations,name='conv2d_%d'%layer_index)
        else:
            r = op.conv2d(r, out_channel=out_dim, filter_size=filter_size, stride=1, activation=tf.nn.relu, padding='VALID', dilations=dilation, name='conv2d_%d'%layer_index)
        index = layer_index + 1
        r = tf.pad(r, [[0,0],[pad,pad],[pad,pad],[0,0]], 'SYMMETRIC')
        r = op.conv2d(r, out_channel=out_dim, filter_size=filter_size, stride=1, activation=None, padidng='VALID',  dilations=dilations, name='conv2d_%d'%layer_index)
        index += 1
        return tf.nn.relu(r+x), index



