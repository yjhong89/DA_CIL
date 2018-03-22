import tensorflow as tf
import numpy as np
import op
import utils
import argparse
import dataset_utils

class generator(object):
    def __init__(self, channel):
        self.channel = channel
        self.module_name = 'generator'

    # Build model
    # From image-to-image translation
    def generator_unet(self, x, name, reuse=False):
        layer_index = 0
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name + '_UNET'):
                if reuse:
                    tf.get_variable_scope().resue_variables()
                
                # All relus in the encoder are leaky with 0.2
                with tf.variable_scope('encoder'):
                    e1 = op.conv2d(x, out_channel=self.channel, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    e2 = op.conv2d(e1, out_channel=self.channel*2, name='conv2d_%d'%layer_index)   
                    layer_index += 1
                    e3 = op.conv2d(e2, out_channel=self.channel*4, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    e4 = op.conv2d(e3, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    e5 = op.conv2d(e4, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    e6 = op.conv2d(e5, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    e7 = op.conv2d(e6, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                    layer_index += 1
                    # Middle point of total architecture(number of total layers=16)
                    e8 = op.conv2d(e7, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                    layer_index += 1
                
                # U-Net architecture is with skip connections between each layer i in the encoer and layer n-i in the decoder. Concatenate activations in channel axis
                # Dropout with 0.5
                with tf.variable_scope('decoder'):
                    d1 = op.transpose_conv2d(e8, out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index)
                    d1 = tf.concat([d1, e7], axis=3)
                    layer_index += 1
                    d2 = op.transpose_conv2d(d1, out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index)
                    d2 = tf.concat([d2, e6], axis=3)
                    layer_index += 1
                    d3 = op.transpose_conv2d(d2, out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index)
                    d3 = tf.concat([d3, e5], axis=3)
                    layer_index += 1
                    d4 = op.transpose_conv2d(d3, out_channel=self.channel*8, name='transpose_conv2d_%d'%layer_index)
                    d4 = tf.concat([d4, e4], axis=3)
                    layer_index += 1
                    d5 = op.transpose_conv2d(d4, out_channel=self.channel*4, name='transpose_conv2d_%d'%layer_index)
                    d5 = tf.concat([d5, e3], axis=3)
                    layer_index += 1
                    d6 = op.transpose_conv2d(d5, out_channel=self.channel*2, name='transpose_conv2d_%d'%layer_index)
                    d6 = tf.concat([d6, e2], axis=3)
                    layer_index += 1
                    d7 = op.transpose_conv2d(d6, out_channel=self.channel, name='transpose_conv2d_%d'%layer_index)
                    d7 = tf.concat([d7, e2], axis=3)
                    layer_index += 1
                    d8 = op.transpose_conv2d(d7, out_channel=3, name='transpose_conv2d_%d'%layer_index)
                
        return tf.nn.tanh(d8) 


    # Dilated Residual Networks
    def generator_drn(self, x, name, reuse=False):
        layer_index = 0
        residual_index = 0
        with tf.variable_scope(self.module_name):
            with tf.variable_scope(name+'_DRN'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
    
                # image_size - fiter_size + 2*pad + 1 (when stride=1)
                x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'SYMMETRIC')
                x = op.conv2d(x, out_channel=int(self.channel//4), filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index)
                layer_index += 1
                x, layer_index = op.residual_block(x, out_dim=int(self.channel//4), layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=int(self.channel//2), layer_index=layer_index, downsample=True, name='residual_%d'%residual_index)
                residual_index += 1
                 
                x, layer_index = op.residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
                residual_index += 1
    
                # Removing gridding artifacts
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, dilations=[1,2,2,1],name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, dilations=[1,2,2,1],name='conv2d_%d'%layer_index)
                layer_index += 1
                
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index)
                layer_index += 1
               
                # Upsampling 
                x = op.transpose_conv2d(x, out_channel=self.channel*4, filter_size=3, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = op.transpose_conv2d(x, out_channel=self.channel*2, filter_size=3, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = op.transpose_conv2d(x, out_channel=self.channel, filter_size=3, name='transpose_conv2d_%d'%layer_index)
                layer_index += 1
                x = op.transpose_conv2d(x, out_channel=3, filter_size=7, stride=1, name='transpose_conv2d_%d'%layer_index)

        return tf.nn.tanh(x)



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
                x = op.conv2d(x, out_channel=self.channel*8, name='conv2d_%d'%layer_index)
                layer_index += 1
                # After the last layer, a convolution is applied to map to a 1 dimensional output
                x = op.conv2d(x, out_channel=1, stride=1, name='conv2d_%d'%layer_index)

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
        with tf.variable_scope(self.module_name):
            # First few layers in the classifier
            with tf.variable_scope(private, reuse=reuse_private):
                # image_size - fiter_size + 2*pad + 1 (when stride=1)
                x = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]], 'SYMMETRIC')
                x = op.conv2d(x, out_channel=self.channel//4, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', normalization=op._batch_norm, name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel//4, layer_index=layer_index, downsample=False, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=self.channel//2, layer_index=layer_index, downsample=True, normalization=op._batch_norm, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                 
                x, layer_index = op.residual_block(x, out_dim=self.channel, layer_index=layer_index, downsample=True, normalizationname='residual_%d'%residual_index, training=self.training)
                residual_index += 1

                x, layer_index = op.residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
    
            # Last layers in the classifier
            with tf.variable_scope(share, reuse=reuse_share):
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*4, dilation_rate=2, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
    
                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, training=self.training)
                residual_index += 1
                x, layer_index = op.residual_block(x, out_dim=self.channel*8, dilation_rate=4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index, training=self.training)

                # Removing gridding artifacts
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, dilations=[1,2,2,1],name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, dilations=[1,2,2,1],name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index, training=self.training)
                layer_index += 1
                x = op.conv2d(x, out_channel=self.channel*8, filter_size=3, stride=1, activation=tf.nn.relu, name='conv2d_%d'%layer_index, training=self.training)
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
    x = tf.get_variable('test', [10, 320,180,3])
    g.generator_drn(x, name='g')
            