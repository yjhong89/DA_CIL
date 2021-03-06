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
        
        return scale * normalized + shift

def _group_norm(x, training=True, name='group_norm', epsilon=1e-5, group_size=32):
    batch_size, height, width, channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [channel], initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift', [channel], initializer=tf.constant_initializer(0))

        # Transpose [N,H,W,C] to [N,C,H,W]
        x_trans = tf.transpose(x, [0,3,1,2])
        group_size = min(group_size, channel)
        x = tf.reshape(x_trans, [batch_size, group_size, channel // group_size, height, width])

        group_mean, group_var = tf.nn.moments(x, [2,3,4], keep_dims=True)

        normalized = (x -group_mean) / tf.sqrt(group_var + epsilon)

        normalized = tf.reshape(normalized, [batch_size, channel, height, width])
        scale = tf.reshape(scale, [1, channel, 1, 1])
        shift = tf.reshape(shift, [1, channel, 1, 1])

        output = scale * normalized + shift
        output = tf.transpose(output, [0,2,3,1])

        return output
    

# From progressive GAN
def _pixel_norm(x, name='pixel_norm', epsilon=1e-8, training=True):
    # NHWC
    with tf.variable_scope(name):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keep_dims=True) + epsilon)


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

# From Attention U-NET
def attention_gate(x, g, f_int, layer_index, name='attention', normalization=_instance_norm):
    index = layer_index
    with tf.name_scope(name):       
        gate_feature = conv2d(g, out_channel=f_int, filter_size=1, stride=1, name='gate_%d'%index, activation=None, normalization=None, bias=False)
        input_feature = conv2d(x, out_channel=f_int, filter_size=1, stride=1, name='input_%d'%index, activation=None, normalization=None, bias=True)
        '''
            gate_feature, input_feature has different dimension : Downsample to the resolution of gating signal
            gate_feature: [batch size, h_g, w_g, f_int]
            input_feature: [batch size, h_g*2, w_g*2, f_int]
        '''
        input_feature_downsampled = tf.image.resize_images(input_feature, gate_feature.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR)
        attention_signal = tf.nn.relu(gate_feature + input_feature_downsampled)
        attention_signal = conv2d(attention_signal, out_channel=1, filter_size=1, stride=1, name='attention_%d'%index, activation=tf.nn.sigmoid, normalization=None, bias=True)
        # Upsample to input 'x' shape
        additive_attention = tf.image.resize_images(attention_signal, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR)
        
        result = additive_attention * x

        # Output transform
        result = conv2d(result, out_channel=x.get_shape().as_list()[-1], filter_size=1, stride=1, name='output_transform_%d'%layer_index, activation=None, normalization=normalization, bias=False)

    return result


def residual_block(x, out_dim, layer_index, filter_size=3, stride=1, name='residual', normalization=_instance_norm, downsample=True, training=True, activation=tf.nn.relu):
    in_dim = x.get_shape().as_list()[-1]
    if in_dim == out_dim:
        increase_dim = False
    else:
        increase_dim = True

    with tf.name_scope(name):
        padding = int((filter_size - 1) / 2)
        if downsample:
            x = _max_pool(x)
            y = tf.pad(x, [[0,0], [padding, padding], [padding, padding], [0,0]], 'REFLECT')
            y = conv2d(y, out_channel=out_dim, filter_size=filter_size, stride=stride, name='conv2d_%d'%layer_index, activation=activation, padding='VALID', normalization=normalization, training=training)
            x = tf.pad(x, [[0,0],[0,0],[0,0],[in_dim//2,in_dim//2]], 'CONSTANT')

        else:
            if increase_dim:
                x = tf.pad(x, [[0,0],[0,0],[0,0],[in_dim//2,in_dim//2]], 'CONSTANT')
            
            y = tf.pad(x, [[0,0],[padding, padding], [padding,padding],[0,0]], 'REFLECT')
            y = conv2d(y, out_channel=out_dim, filter_size=filter_size, stride=stride, name='conv2d_%d'%layer_index, activation=activation, padding='VALID', normalization=normalization, training=training)
        layer_index += 1

        y = tf.pad(y, [[0,0],[padding, padding], [padding,padding],[0,0]], 'REFLECT')
        y = conv2d(y, out_channel=out_dim, filter_size=filter_size, stride=stride, name='conv2d_%d'%layer_index, activation=False, padding='VALID', normalization=normalization, training=training)
        layer_index += 1

    return x + y, layer_index

def fc(x, hidden, dropout_ratio=0.5, activation=tf.nn.relu, dropout=True, name='fc', bias=True):
    _, in_dim = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[in_dim, hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        output = tf.matmul(x, weight)
        if bias:
            bias = tf.get_variable('bias', shape=[hidden], initializer=tf.constant_initializer(0.1))
            output = output + bias

        if dropout:
            output = tf.nn.dropout(output, dropout_ratio, name='dropout')
        if activation:
            output = activation(output)

    return output

def global_average_pooling(x, name='global_pooling'):
    # Make fully connected layer
    return tf.reduce_mean(x, [1,2])
    
def dilated_conv2d(x, out_channel, filter_size, dilation_rate, name='dilated_conv2d', activation=tf.nn.relu, normalization=_instance_norm, padding='VALID', training=True, bias=True):
    _, _, _, in_channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
                
        rate = [dilation_rate, dilation_rate]
        # stride must be 1
        output = tf.nn.convolution(x, weight, padding=padding, dilation_rate=rate)

        if bias:
            bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0.1))
            output = output + bias

        if normalization:
            output = normalization(output)
        if activation:
            output = activation(output)

        #print('dilated convolution', output.get_shape().as_list())

    return output      


def conv2d(x, out_channel, filter_size=4, stride=2, name='conv2d', activation=_leaky_relu, normalization=_instance_norm, padding='SAME', training=True, bias=True):
    _, _, _, in_channel = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # NHWC: batch, height, width, channel
        # padding VALID: no zero padding, only covers the valid input
        output = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding=padding, data_format='NHWC', name='convolution')
        
        if bias:
            bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0.1))
            output = output + bias

        if normalization:
            output = normalization(output, training=training)
        if activation:
            output = activation(output)

        #print('convolution', output.get_shape().as_list())

    return output

# size_out = stride*(size_in) + filter_size - 2*pad
# Literally transposed version of convolution
def transpose_conv2d(x, out_channel, filter_size=4, stride=2, name='transpose_conv2d', activation=tf.nn.relu, normalization=_instance_norm, dropout=False, dropout_rate=0.5):
    batch_size, height, width, in_channel = x.get_shape().as_list()
    new_height, new_width = int(height*stride), int(width*stride)
    out_shape = tf.stack([batch_size, new_height, new_width, out_channel])
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [filter_size, filter_size, out_channel, in_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [out_channel], initializer=tf.constant_initializer(0))

        output = tf.nn.conv2d_transpose(x, weight, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME', data_format='NHWC', name='transposed_convolution')
        output = output + bias

        if dropout:
            output = tf.nn.dropout(output, dropout_rate)
        if normalization:
            output = normalization(output)
        if activation:
            output = activation(output)
        #print('upsampling', output.get_shape().as_list())
            
        return output        

# To downsize width and height when downsampling
def _max_pool(x, kernel=[1,2,2,1], stride=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding='SAME')
        

def dilated_residual_block(x, out_dim, layer_index, dilation_rate=1, filter_size=3, downsample=True, name='residual', normalization=_instance_norm, training=True):
    in_dim = x.get_shape().as_list()[-1]
    if in_dim * 2 == out_dim:
        increase_dim = True
    else:
        increase_dim = False

    with tf.variable_scope(name):
        # int: floor
        pad = int((filter_size - 1) / 2 * dilation_rate)
        r = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], 'REFLECT')
        # Note that when using dilated convolution, stride must be 1
        # When down sampling, [height, width, channel] -> [height/2, width/2, channel*2], so conv2 and x does not match
        # From Resnet, we add zero pads to increase the depth
        if downsample:
            x = _max_pool(x)
            r = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], 'REFLECT')
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
        r1 = tf.pad(r1, [[0,0],[pad,pad],[pad,pad],[0,0]], 'REFLECT')
        r2 = dilated_conv2d(r1, out_channel=out_dim, filter_size=filter_size, activation=None, padding='VALID', dilation_rate=dilation_rate, name='conv2d_%d'%index, normalization=normalization, training=training)
        #print('layer: %d, shape of x: %s, shape of r: %s' % (layer_index, x.get_shape().as_list(), r2.get_shape().as_list()))
        index += 1

        # Do not use non-linear activation
        return r2+x, index



