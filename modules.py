import tensorflow as tf
import numpy as np
import op
import utils
import argparse
import dataset_utils

class generator(object):
    def __init__(self, channel, config, batch_size):
        self.channel = channel
        self.module_name = 'generator'
        self.config = config

        self.latent_vars = dict()
        # Add latent noise vector to image input
        if self.config.getboolean(self.module_name, 'noise'):
            tf.logging.info('Using random noise')
            noise = tf.random_uniform(shape=[batch_size, self.config.getint(self.module_name, 'noise_dim')], minval=-1, maxval=1, dtype=tf.float32, name='random_noise')
            self.latent_vars['noise'] = noise


    # Build model
    # From image-to-image translation
    def generator_unet(self, x, name, reuse=False):
        layer_index = 0
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
                    e1 = op.conv2d(x, out_channel=self.channel, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    e2 = op.conv2d(op._leaky_relu(e1), out_channel=self.channel*2, name='conv2d_%d'%layer_index, activation=False)   
                    layer_index += 1
                    e3 = op.conv2d(op._leaky_relu(e2), out_channel=self.channel*4, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    e4 = op.conv2d(op._leaky_relu(e3), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    e5 = op.conv2d(op._leaky_relu(e4), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    e6 = op.conv2d(op._leaky_relu(e5), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    e7 = op.conv2d(op._leaky_relu(e6), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                    # Middle point of total architecture(number of total layers=16)
                    e8 = op.conv2d(op._leaky_relu(e7), out_channel=self.channel*8, name='conv2d_%d'%layer_index, activation=False)
                    layer_index += 1
                
                # U-Net architecture is with skip connections between each layer i in the encoer and layer n-i in the decoder. Concatenate activations in channel axis
                # Dropout with 0.5
                with tf.variable_scope('decoder'):
                    d1 = op.transpose_conv2d(tf.nn.relu(e8), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d1 = tf.concat([d1, e7], axis=3)
                    layer_index += 1
                    d2 = op.transpose_conv2d(tf.nn.relu(d1), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d2 = tf.concat([d2, e6], axis=3)
                    layer_index += 1
                    d3 = op.transpose_conv2d(tf.nn.relu(d2), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d3 = tf.concat([d3, e5], axis=3)
                    layer_index += 1
                    d4 = op.transpose_conv2d(tf.nn.relu(d3), out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d4 = tf.concat([d4, e4], axis=3)
                    layer_index += 1
                    d5 = op.transpose_conv2d(tf.nn.relu(d4), out_channel=self.channel*4, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d5 = tf.concat([d5, e3], axis=3)
                    layer_index += 1
                    d6 = op.transpose_conv2d(tf.nn.relu(d5), out_channel=self.channel*2, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d6 = tf.concat([d6, e2], axis=3)
                    layer_index += 1
                    d7 = op.transpose_conv2d(tf.nn.relu(d6), out_channel=self.channel, name='transpose_conv2d_%d'%layer_index, activation=False)
                    d7 = tf.concat([d7, e1], axis=3)
                    layer_index += 1
                    d8 = op.transpose_conv2d(tf.nn.relu(d7), out_channel=3, name='transpose_conv2d_%d'%layer_index, normalization=False, activation=tf.nn.tanh)
                
        return d8 


    # Dilated Residual Networks
    def generator_drn(self, x, name, reuse=False):
        layer_index = 0
        residual_index = 0
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
                x = op.conv2d(x, out_channel=self.channel, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index)
                layer_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                layer_index += 1

                # Down sample
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1

                # Dilation instead of down sample
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.dilated_residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
                 
#                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
#                residual_index += 1
#                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
#                residual_index += 1
#
#                x = op.dilated_conv2d(x, out_channel=self.channel*4, filter_size=3, activation=tf.nn.relu, dilation_rate=2, name='conv2d_%d'%layer_index, padding='SAME')
#                layer_index += 1
#                x = op.dilated_conv2d(x, out_channel=self.channel*4, filter_size=3, activation=tf.nn.relu, dilation_rate=2, name='conv2d_%d'%layer_index, padding='SAME')
#                layer_index += 1

                # Removing gridding artifacts
                x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*2, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1

                # Upsampling 
                x = op.transpose_conv2d(x, out_channel=self.channel, filter_size=3, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                x = op.conv2d(x, out_channel=3, filter_size=7, stride=1, padding='VALID', name='transpose_conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                

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
                x = op.conv2d(x, out_channel=self.channel, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index)
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
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, layer_index=layer_index, name='residual_%d'%residual_index)
                residual_index += 1

                # Upsampling 
                x = op.transpose_conv2d(x, out_channel=self.channel*2, filter_size=3, stride=2, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = op.transpose_conv2d(x, out_channel=self.channel, filter_size=3, stride=2, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                x = op.conv2d(x, out_channel=3, filter_size=7, stride=1, padding='VALID', name='transpose_conv2d_%d'%layer_index, normalization=None, activation=tf.nn.tanh)
                
        return x


def project_latent_vars(shape, latent_vars, combine_method, name):
    values = list()
    # Keys
    for var in latent_vars:
        with tf.variable_scope(var):
            # Project and reshape noise to NHWC
            projected = op.fc(latent_vars[var], np.prod(shape[1:]), dropout=False, name=name+var)
        values.append(tf.reshape(projected, [shape[0]] + shape[1:]))

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
    def __init__(self, channel):
        self.channel = channel
        self.module_name = 'discriminator'

    # 70x70 PatchGAN to model high frequency region
    # why 70x70?
        # regular GAN discriminaotr maps image to a single scalar while patchGAN maps image to an NXN array of output X
        # X_ij signifies patch_ij in input image is real or fake. -> so 70x70 patches in input images
        # equivalent manually chopped up the image into 70x70 patch, run a regular discriminator
    def __call__(self, x, name, reuse=False):
        layer_index = 0
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
    
                # From cycleGAN, do not use instance Norm for the first C64 layer
                x = op.conv2d(x, out_channel=self.channel, normalization=False, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*2, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*4, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, stride=1, name='conv2d_%d'%layer_index)
                layer_index += 1
                # After the last layer, a convolution is applied to map to a 1 dimensional output
                x = op.conv2d(x, out_channel=1, stride=1, name='conv2d_%d'%layer_index, activation=None, normalization=False)

        return x

class task_regression(object):
    def __init__(self, channel, image_fc, measurement_fc, command_fc, dropout, name='task_regression'):
        self.channel = channel
        self.name = name
        self.image_fc = image_fc
        self.measurement_fc = measurement_fc
        self.command_fc = command_fc
        self.num_command = 4
        self.dropout = dropout
        # To optimize jointly with disicriminator since task regression and disicriminator are not related
        self.module_name = 'disciminator'


    def __call__(self, image, measurements, reuse=False):
        image_layer_index = 0
        measurement_layer_index = 0
        branches = list()
        
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(self.name):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope('image_module'):
                    x = op.conv2d(image, out_channel=self.channel, filter_size=5, stride=2, normalization=False, activation=tf.nn.relu, name='conv2d_%d'%image_layer_index)
                    image_layer_index += 1
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
                    x = op.fc(flatten, self.image_fc, dropout_ratio=self.dropout, name='fc_%d'%image_layer_index)
                    image_layer_index += 1
                    x = op.fc(x, self.image_fc, dropout_ratio=self.dropout, name='fc_%d'%image_layer_index)
    
                with tf.variable_scope('measurement_module'):
                    y = op.fc(measurements, self.measurement_fc, dropout_ratio=self.dropout, name='fc_%d'%measurement_layer_index)
                    measurement_layer_index += 1    
                    y = op.fc(y, self.measurement_fc, dropout_ratio=self.dropout, name='fc_%d'%measurement_layer_index)
                    
                with tf.variable_scope('joint'):
                    joint = tf.concat([x,y], axis=-1, name='joint_representation')
                    joint = op.fc(joint, self.image_fc, dropout_ratio=self.dropout, name='fc')
        
                for i in len(self.num_commands):
                    branch_layer_index = 0
                    with tf.variable_scope('branch_%d'%i):
                        branch_output = op.fc(joint, self.branch_fc, dropout_ratio=self.dropout, name='fc_%d'%branch_layer_index)
                        branch_layer_index += 1
                        branch_output = op.fc(branch_output, self.branch_fc, dropout_ratio=self.dropout, name='fc_%d'%branch_layer_index)
                        branch_layer_index += 1
                        branch_output = op.fc(branch_output, len(self.num_output), dropout_ratio=self.dropout, name='fc_%d'%branch_layer_index)
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
            
