import tensorflow as tf
import numpy as np
import op
import utils
import argparse
import dataset_utils

class generator(object):
    def __init__(self, args, name='G_S_to_T'):
        self.args = args
        self.name = name
        
        section = 'generator'
        config = utils.MyConfigParser()
        utils.load_config(config, 'config.ini')
        self.channel = config.getint(section, 'channel')
        
    # Build model
    # From image-to-image translation
    def generator_unet(self, x):
        layer_index = 0
        with tf.variable_scope(self.name + '_UNET'):
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
    def generator_drn(self, x):
        layer_index = 0
        residual_index = 0
        with tf.variable_scope(self.name+'_DRN'):
            # image_size - fiter_size + 2*pad + 1 (when stride=1)
            x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'SYMMETRIC')
            x = op.conv2d(x, out_channel=self.channel//4, filter_size=7, stride=1, activation=tf.nn.relu, padding='VALID', name='conv2d_%d'%layer_index)
            layer_index += 1
            x, layer_index = op.residual_block(x, out_dim=self.channel//4, layer_index=layer_index, downsample=False, name='residual_%d'%residual_index)
            residual_index += 1

            x, layer_index = op.residual_block(x, out_dim=self.channel//2, layer_index=layer_index, downsample=True, name='residual_%d'%residual_index)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    g = generator(args)
    x = tf.get_variable('test', [10, 224,224,3])
    g.generator_drn(x)
            



