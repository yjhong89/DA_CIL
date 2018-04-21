import tensorflow as tf
import numpy as np
import op
import utils
import argparse
import dataset_utils
import tensorflow.contrib.slim as slim 


class generator(object):
    def __init__(self, channel, config, args):
        self.channel = channel
        self.module_name = 'generator'
        self.config = config
        self.args = args

        self.out_channel = self.config.getint(self.module_name, 'out_channel')
        self.latent_vars = dict()
        # Add latent noise vector to image input
        if self.config.getboolean(self.module_name, 'noise'):
            tf.logging.info('Using random noise')
            noise = tf.random_uniform(shape=[self.args.batch_size, self.config.getint(self.module_name, 'noise_dim')], minval=-1, maxval=1, dtype=tf.float32, name='random_noise')
            self.latent_vars['noise'] = noise
        elif(self.out_channel != 3):
            raise ValueError('No random noise injection')

    def normalize(self):
        return op._pixel_norm if self.args.pixel_norm else op._instance_norm

    # Build model
    # From image-to-image translation
    def generator_unet(self, x, name, reuse=False):
        layer_index = 0
        normalize_func = self.normalize()
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name + '_UNET'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                
                if self.latent_vars:
                    noise_channel = project_latent_vars(shape=x.get_shape().as_list()[0:3]+[1],
                                        latent_vars=self.latent_vars,
                                        combine_method='concat', name=name)
                    x = tf.concat([x, noise_channel], axis=3)

                with tf.variable_scope('encoder'):
                    e1 = op.conv2d(x, out_channel=self.channel, name='conv2d_%d'%layer_index, normalization=False, activation=False)
                    layer_index += 1
                    e2 = op.conv2d(tf.nn.relu(e1), out_channel=self.channel*2, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)   
                    layer_index += 1
                    e3 = op.conv2d(tf.nn.relu(e2), out_channel=self.channel*4, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                    e4 = op.conv2d(tf.nn.relu(e3), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                    e5 = op.conv2d(tf.nn.relu(e4), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                    e6 = op.conv2d(tf.nn.relu(e5), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                    e7 = op.conv2d(tf.nn.relu(e6), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                    # Middle point of total architecture(number of total layers=16)
                    e8 = op.conv2d(tf.nn.relu(e7), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    layer_index += 1
                
                # U-Net architecture is with skip connections between each layer i in the encoer and layer n-i in the decoder. Concatenate activations in channel axis
                # Dropout with 0.5
                with tf.variable_scope('decoder'):
                    d1 = op.transpose_conv2d(tf.nn.relu(e8), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False, dropout=True, normalization=normalize_func)
                    d1 = tf.concat([d1, e7], axis=3)
                    layer_index += 1
                    d2 = op.transpose_conv2d(tf.nn.relu(d1), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False, dropout=True, normalization=normalize_func)
                    d2 = tf.concat([d2, e6], axis=3)
                    layer_index += 1
                    d3 = op.transpose_conv2d(tf.nn.relu(d2), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False, dropout=True, normalization=normalize_func)
                    d3 = tf.concat([d3, e5], axis=3)
                    layer_index += 1
                    d4 = op.transpose_conv2d(tf.nn.relu(d3), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    d4 = tf.concat([d4, e4], axis=3)
                    layer_index += 1
                    d5 = op.transpose_conv2d(tf.nn.relu(d4), out_channel=self.channel*4, name='transpose_conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    d5 = tf.concat([d5, e3], axis=3)
                    layer_index += 1
                    d6 = op.transpose_conv2d(tf.nn.relu(d5), out_channel=self.channel*2, name='transpose_conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    d6 = tf.concat([d6, e2], axis=3)
                    layer_index += 1
                    d7 = op.transpose_conv2d(tf.nn.relu(d6), out_channel=self.channel, name='transpose_conv2d_%d'%layer_index, activation=False, normalization=normalize_func)
                    d7 = tf.concat([d7, e1], axis=3)
                    layer_index += 1
                    d8 = op.transpose_conv2d(tf.nn.relu(d7), out_channel=self.out_channel, name='transpose_conv2d_%d'%layer_index, normalization=False, activation=tf.nn.tanh)
                
        return d8, noise_channel


    # Dilated Residual Networks
    def generator_drn(self, x, name, reuse=False):
        layer_index = 0
        residual_index = 0
        normalize_func = self.normalize()
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name+'_DRN'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                if self.latent_vars:
                    noise_channel = project_latent_vars(shape=x.get_shape().as_list()[0:3]+[1],
                                        latent_vars=self.latent_vars,
                                        combine_method='concat', name=name)
                    x = tf.concat([x, noise_channel], axis=3)

                # image_size - fiter_size + 2*pad + 1 (when stride=1)
                x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                x = op.conv2d(x, out_channel=self.channel, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index, normalization=False)
                layer_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, normalization=normalize_func)
                layer_index += 1

                # Down sample
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index, normalization=normalize_func)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, normalization=normalize_func)
                residual_index += 1

                # Dilation instead of down sample
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, normalization=normalize_func)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, normalization=normalize_func)
                residual_index += 1
                 
#                x = op.dilated_conv2d(x, out_channel=self.channel*4, filter_size=3, activation=tf.nn.relu, dilation_rate=2, name='conv2d_%d'%layer_index, padding='SAME')
#                layer_index += 1
#                x = op.dilated_conv2d(x, out_channel=self.channel*4, filter_size=3, activation=tf.nn.relu, dilation_rate=2, name='conv2d_%d'%layer_index, padding='SAME')
#                layer_index += 1

                # Removing gridding artifacts
                x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index, normalization=normalize_func)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index, normalization=normalize_func)
                layer_index += 1

                # Upsampling 
                x = op.transpose_conv2d(x, out_channel=self.channel, filter_size=3, name='transpose_conv2d_%d'%layer_index, normalization=normalize_func)
                layer_index += 1
                #x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                #x = op.conv2d(x, out_channel=3, filter_size=7, stride=1, padding='VALID', name='transpose_conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                x = op.transpose_conv2d(x, out_channel=self.out_channel, filter_size=7, stride=1, name='transpose_conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                 

        return x

    # Justin Johnson`s model with 9 blocks
    def generator_resnet(self, x, name, reuse=False):
        layer_index = 0
        residual_index = 0
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name+'_RESNET'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                if self.latent_vars:
                    noise_channel = project_latent_vars(shape=x.get_shape().as_list()[0:3]+[1],
                                        latent_vars=self.latent_vars,
                                        combine_method='concat', name=name)
                    x = tf.concat([x, noise_channel], axis=3)

                # image_size - fiter_size + 2*pad + 1 (when stride=1)
                x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                x = op.conv2d(x, out_channel=self.channel, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index, normalization=False)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=2, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*4, filter_size=3, stride=2, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1

                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                layer_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
               # x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                #residual_index += 1
                #x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                #residual_index += 1
                #x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                #residual_index += 1

                # Upsampling 
                x = op.transpose_conv2d(x, out_channel=self.channel*2, filter_size=3, stride=2, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = op.transpose_conv2d(x, out_channel=self.channel, filter_size=3, stride=2, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                #x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                #x = op.conv2d(x, out_channel=3, filter_size=7, stride=1, padding='VALID', name='conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                #x = op.conv2d(x, out_channel=3, filter_size=1, stride=1, padding='SAME', name='conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                x = op.transpose_conv2d(x, out_channel=self.out_channel, filter_size=7, stride=1, name='transpose_conv2d_%d'%layer_index, normalization=False, activation=tf.nn.tanh)
                
        return x, noise_channel


def project_latent_vars(shape, latent_vars, combine_method, name):
    values = list()
    # Keys
    for var in latent_vars:
        with tf.variable_scope(var):
            # Project and reshape noise to NHWC
            projected = op.fc(latent_vars[var], np.prod(shape[1:]), dropout=False, name=name+var)
        values.append(tf.reshape(projected, [shape[0]] + shape[1:]))

    if combine_method == 'sum':
        result = values[0]
        for value in values[1:]:
            result += value
    elif combine_method == 'concat':
        # Concatenate along last axis
        result = tf.concat(values, axis=3)
    else:
        raise ValueError('Unknown combine method %s' % combine_method)

    tf.logging.info('Latent variables for %s projected to size %s volume' % (name, result.shape))

    return result


class discriminator(object):
    def __init__(self, channel, group_size=8):
        self.channel = channel
        self.module_name = 'discriminator'
        self.group_size = group_size

    # 70x70 PatchGAN to model high frequency region
    # why 70x70?
        # regular GAN discriminaotr maps image to a single scalar while patchGAN maps image to an NXN array of output X
        # X_ij signifies patch_ij in input image is real or fake. -> so 70x70 patches in input images
        # equivalent manually chopped up the image into 70x70 patch, run a regular discriminator
    def __call__(self, x, name, patch=True, reuse=False, dropout_prob=0.35, training=True):
        layer_index = 0
        if training:
            self.dropout_prob = dropout_prob
        else:
            self.dropout_prob = 1.0
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name):
                # Add optional noise
                def add_noise(hidden, scope_num=None):
                    if scope_num:
                        hidden = slim.dropout(hidden, self.dropout_prob, is_training=training, scope='dropout_%s' % scope_num)
                    return hidden + tf.random_normal(hidden.shape.as_list(), mean=0, stddev=0.2)


                if reuse:
                    tf.get_variable_scope().reuse_variables()
    
                # From cycleGAN, do not use instance Norm for the first C64 layer
                x = op.conv2d(x, out_channel=self.channel, normalization=False, name='conv2d_%d'%layer_index)
                x = add_noise(x, layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*2, name='conv2d_%d'%layer_index)
                x = add_noise(x, layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*4, name='conv2d_%d'%layer_index)
                x = add_noise(x, layer_index)
                layer_index += 1
                if self.group_size > 1:
                    x = self.minibatch_discrimination(x, self.group_size)
                x = op.conv2d(x, out_channel=self.channel*8, stride=1, name='conv2d_%d'%layer_index)
                x = add_noise(x, layer_index)
                layer_index += 1

                if patch:
                    # After the last layer, a convolution is applied to map to a 1 dimensional output
                    x = op.conv2d(x, out_channel=1, stride=1, name='conv2d_%d'%layer_index, activation=None, normalization=False)
                else:
                    x = slim.flatten(x)
                    x = op.fc(x, 1, activation=None, dropout=False, name='fc')

        return x

    # From progressive GAN
    def minibatch_discrimination(self, x, group_size=4):
        batch_size, height, width, channel = x.get_shape().as_list()
        with tf.variable_scope('minibatch_discrimination'):
            group_size = tf.minimum(group_size, batch_size)
            # Batch size must be divisible by group size
                # [G, M, H, W, C]
            y = tf.reshape(x, [group_size, -1, height, width, channel])
            # First compute the standard deviation for each feature in each spatial location over the minibatch
                # If group_size=1, y - mean would be zero
                # For full minibatch discriminator, make group size equal to batch size
                # In here, we do this in terms of group, [G, M, H, W, C]
            y = y - tf.reduce_mean(y, axis=0, keep_dims=True)
            # Calculate variance and then standard deviation
                # [M, H, W, C]
            y = tf.sqrt(tf.reduce_mean(tf.square(y), axis=0) + 1e-8)
            # Average these estimates over all featuers and spatial locations to arrive at a single value
                # [M, 1, 1, 1]
            y = tf.reduce_mean(y, axis=[1,2,3], keep_dims=True)
            # Replicate the value 
            y = tf.tile(y, [group_size, height, width, 1])
        # Concatenate it to all spatial locations and over the minibatch, yielding one additional feature map
        return tf.concat([x, y], axis=3)



class task_regression(object):
    def __init__(self, channel, image_fc, measurement_fc, command_fc, name='task_regression', training=True):
        self.channel = channel
        self.name = name
        self.image_fc = image_fc
        self.measurement_fc = measurement_fc
        self.command_fc = command_fc
        self.num_command = 4
        self.training = training    
        if self.training:
            self.dropout = [1.0] * 2
        else:
            self.dropout = [0.5, 0.2]

    
        # To optimize jointly with disicriminator since task regression and disicriminator are not related
        self.module_name = 'disciminator'


    def __call__(self, image, measurements, reuse_private=False, reuse_shared=False, private='private', shared='shared_task', mode='RESNET'):
        image_layer_index = 0
        measurement_layer_index = 0
        
        branches = list()
        
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(self.name):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope('image_module'):
                    if mode == 'RESNET':
                        with tf.variable_scope(private, reuse_private):
                            x = op.conv2d(image, out_channel=self.channel, filter_size=7, stride=2, normalization=op._batch_norm, activation=tf.nn.relu, name='conv2d_%d'image_layer_index)
                        image_layer_index += 1
    
                        with tf.variable_scope(shared, reuse_shared):
                            x, image_layer_index = residual_block(x, out_channel=self.channel, image_layer_index, normalization=op._batch_norm, downsample=False, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel, image_layer_index, normalization=op._batch_norm, downsample=False, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*2, image_layer_index, normalization=op._batch_norm, downsample=True, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*2, image_layer_index, normalization=op._batch_norm, downsample=False, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*4, image_layer_index, normalization=op._batch_norm, downsample=True, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*4, image_layer_index, normalization=op._batch_norm, downsample=False, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*8, image_layer_index, normalization=op._batch_norm, downsample=True, training=self.training)
                            x, image_layer_index = residual_block(x, out_channel=self.channel*8, image_layer_index, normalization=op._batch_norm, downsample=False, training=self.training)
               
                    else:
                        with tf.variable_scope(private, reuse_private):
                            x = op.conv2d(image, out_channel=self.channel, filter_size=5, stride=2, normalization=False, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
                        with tf.variable_scope(shared, reuse_shared):
                            x = op.conv2d(x, out_channel=self.channel, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
        
                            x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=2, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
                            x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
    
                            x = op.conv2d(x, out_channel=self.channel*4, filter_size=3, stride=2, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
                            x = op.conv2d(x, out_channel=self.channel*4, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index) 
                            image_layer_index += 1
                            x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
                            x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                            image_layer_index += 1
   
                    x_shape = x.get_shape().as_list()
                    # Fully connected layer
                    flatten = tf.reshape(x, [-1, x_shape.prod[1:]])
                    x = op.fc(flatten, self.image_fc, dropout_ratio=self.dropout[1], name='fc_%d'%image_layer_index)
                    image_layer_index += 1
                    x = op.fc(x, self.image_fc, dropout_ratio=self.dropout[1], name='fc_%d'%image_layer_index)
    
                with tf.variable_scope('measurement_module'):
                    y = op.fc(measurements, self.measurement_fc, dropout_ratio=self.dropout, name='fc_%d'%measurement_layer_index)
                    measurement_layer_index += 1    
                    y = op.fc(y, self.measurement_fc, dropout_ratio=self.dropout, name='fc_%d'%measurement_layer_index)
                    
                with tf.variable_scope('joint'):
                    joint = tf.concat([x,y], axis=-1, name='joint_representation')
                    joint = op.fc(joint, self.image_fc, dropout_ratio=self.dropout[0], name='fc')
        
                for i in len(self.num_commands):
                    branch_layer_index = 0
                    with tf.variable_scope('branch_%d'%i):
                        branch_output = op.fc(joint, self.branch_fc, dropout_ratio=self.dropout[0], name='fc_%d'%branch_layer_index)
                        branch_layer_index += 1
                        branch_output = op.fc(branch_output, self.branch_fc, dropout_ratio=self.dropout[0], name='fc_%d'%branch_layer_index)
                        branch_layer_index += 1
                        branch_output = op.fc(branch_output, len(self.num_output), dropout_ratio=self.dropout[0], name='fc_%d'%branch_layer_index)
                        branches.append(branch_output)
    
        return branches
        
class task_classifier(object):
    def __init__(self, channel, num_classes, training=True):
        self.channel = channel
        self.num_classes = num_classes
        self.training = training
        
        self.module_name = 'discriminator'
        
    # Based on dilated Residual Network and Unsupervised pixel-lelvel domain adaptation
    def __call__(self, image, reuse_private=False, reuse_shared=False, shared='shared', private='private_task'):
        layer_index = 0
        residual_index = 0
        with tf.variable_scope(self.module_name):
            # First few layers in the classifier
            with tf.variable_scope(private, reuse=reuse_private):
                #print(image.get_shape().as_list())
                # image_size - fiter_size + 2*pad + 1 (when stride=1)
                x = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]], 'SYMMETRIC')
                x = op.conv2d(x, out_channel=self.channel, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', normalization=op._batch_norm, name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=False, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
    
            # Last layers in the classifier
            with tf.variable_scope(shared, reuse=reuse_shared):
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=True, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=False, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1

                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1

                # Removing gridding artifacts
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, normalization=op._batch_norm, name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, normalization=op._batch_norm, name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
               
                # Global average pooling for classification output
                x = op.global_average_pooling(x)

                head_logits = op.fc(x, self.num_classes, dropout=False, activation=None, name='head')
                lateral_logits = op.fc(x, self.num_classes, dropout=False, activation=None, name='latera')

        return head_logits, lateral_logits



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    g = generator(64)
    x = tf.get_variable('test', [10, 256, 256,3])
    #d(x, name='d')
    g.generator_unet(x, name='g')
            
