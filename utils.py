import tensorflow as tf
import numpy as np
import os
import configparser
import dataset_utils
import inspect
import math
import scipy.misc

class MyConfigParser(configparser.ConfigParser):
    def getlist(self, section, option):
        value = self.get(section, option)
        return list(filter(None, value.split(',')))

    def getlistint(self, section, option):
        return [int(x) for x in self.getlist(section, option)]
        
    
def load_config(config, ini):
    config_file = os.path.expanduser(ini)
    config.read(config_file)

def make_savedir(config):
    model_type = config.get('config', 'experiment')
    adversarial_mode = config.get('config', 'mode')
    noise = config.getboolean('generator', 'noise')
    patch = config.getboolean('discriminator', 'patch')
    gen_type = config.get('generator', 'type')
    source_only = config.getboolean('config', 'source_only')
    t2s_task = config.getboolean('config', 't2s_task')
    style_weights = config.getlist('discriminator', 'style_weights')

    result = model_type + '_' + adversarial_mode + '_' + gen_type

    if noise:
        result += '_noise'

    if source_only:
        result = result + '_source_only'

    if t2s_task:
        result += '_t2s_task'

    if sum([float(y) for y in style_weights]) != 0:
        result += '_style'

    if patch:
        result += '_patch'

    return result

def str2bool(v):
    if v.lower() in ('true', 't', 'y', 'yes'):
        return True
    elif v.lower() in ('false', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError('%s is not supported' % v)

def summarize(summary_set, t2s_option, source_only=False):
    _, _, _, source_channels = summary_set['source_image'].get_shape().as_list()
    tf.logging.info('%d channels' % source_channels)

    if source_only:
        task_loss_summary = tf.summary.scalar('task_loss', summary_set['classification_loss'])
        source_image1_summary, source_image2_summary, source_image3_summary = _summarize_transferred_grid(summary_set['source_image'], channels=source_channels)
        return tf.summary.merge([task_loss_summary, source_image1_summary, source_image2_summary, source_image3_summary])

    # Loss part
    source_cyclic_summary = tf.summary.scalar('source_cyclic_loss', summary_set['source_cyclic_loss'])
    target_cyclic_summary = tf.summary.scalar('target_cyclic_loss', summary_set['target_cyclic_loss'])
    s2t_g_loss_summary = tf.summary.scalar('s2t_g_loss', summary_set['s2t_g_loss'])
    t2s_g_loss_summary = tf.summary.scalar('t2s_g_loss', summary_set['t2s_g_loss'])
    s2t_style_loss_summary = tf.summary.scalar('s2t_style_loss', summary_set['s2t_style_loss'])
    t2s_style_loss_summary = tf.summary.scalar('t2s_style_loss', summary_set['t2s_style_loss'])
    s2t_d_loss_summary = tf.summary.scalar('s2t_d_loss', summary_set['s2t_d_loss'])
    t2s_d_loss_summary = tf.summary.scalar('t2s_d_loss', summary_set['t2s_d_loss'])
    task_loss_summary = tf.summary.scalar('task_loss', summary_set['task_loss'])
    if t2s_option:
        t2s_task_loss_summary = tf.summary.scalar('t2s_task_loss', summary_set['t2s_task_loss'])
    else:
        t2s_task_loss_summary = tf.summary.scalar('t2s_task_loss', 0)
    generator_loss_summary = tf.summary.scalar('generator_loss', summary_set['generator_loss'])
    discriminator_loss_summary = tf.summary.scalar('discriminator_loss', summary_set['discriminator_loss'])

    generator_merged = [source_cyclic_summary, target_cyclic_summary, s2t_g_loss_summary, t2s_g_loss_summary, generator_loss_summary]
    
    _, _, _, target_channels = summary_set['target_image'].get_shape().as_list()
    assert source_channels == target_channels

    # Image part
    if source_channels == 3:
        s2t_summary = _summarize_transferred_grid(summary_set['source_image'], summary_set['source_transferred'], name='S2T', channels=source_channels)
        t2s_summary = _summarize_transferred_grid(summary_set['target_image'], summary_set['target_transferred'], name='T2S', channels=target_channels)
        s2t2s_summary = _summarize_transferred_grid(summary_set['source_image'], summary_set['back2source'], name='S2T2S', channels=source_channels)
        t2s2t_summary = _summarize_transferred_grid(summary_set['target_image'], summary_set['back2target'], name='T2S2T', channels=target_channels)
        generator_merged += [s2t_summary, t2s_summary, s2t2s_summary, t2s2t_summary]
    else:
        s2t_summary1, s2t_summary2, s2t_summary3 = _summarize_transferred_grid(summary_set['source_image'], summary_set['source_transferred'], name='S2T', channels=source_channels)
        t2s_summary1, t2s_summary2, t2s_summary3 = _summarize_transferred_grid(summary_set['target_image'], summary_set['target_transferred'], name='T2S', channels=target_channels)
        s2t2s_summary1, s2t2s_summary2, s2t2s_summary3 = _summarize_transferred_grid(summary_set['source_image'], summary_set['back2source'], name='S2T2S', channels=source_channels)
        t2s2t_summary1, t2s2t_summary2, t2s2t_summary3 = _summarize_transferred_grid(summary_set['target_image'], summary_set['back2target'], name='T2S2T', channels=target_channels)
        generator_merged += [s2t_summary1, s2t_summary2, s2t_summary3, t2s_summary1, t2s_summary2, t2s_summary3, s2t2s_summary1, s2t2s_summary2, s2t2s_summary3, t2s2t_summary1, t2s2t_summary2, t2s2t_summary3]

    discriminator_merged = [s2t_d_loss_summary, t2s_d_loss_summary, s2t_style_loss_summary, t2s_style_loss_summary, task_loss_summary, t2s_task_loss_summary, discriminator_loss_summary]

    generator_merge_summary = tf.summary.merge(generator_merged)
    discriminator_merge_summary = tf.summary.merge(discriminator_merged)

    return generator_merge_summary, discriminator_merge_summary
  

def _image_grid(images, max_grid_size=4):
    '''
        images: [batch size, height, width, channels]
        returns max_grid_size*max_grid_size image grid
    '''

    batch_size, height, width, channel = images.get_shape().as_list()
    grid_size = min(int(math.sqrt(batch_size)), max_grid_size)
    
    assert batch_size >= grid_size * grid_size

    if channel != 3:
        # To think easily, assume no channel
        showing_images = images[:grid_size*grid_size,:,:,:3]
        showing_images = tf.reshape(showing_images, [-1, width, 3])
        # tf.split, if num_or_size_split is an integer, split value along axis into num_split tensors
            # if num_or_size_split is not an integer, it is presumed as a split size
        showing_images = tf.split(showing_images, grid_size, 0)
        # [height*grid_size, width*grid_size, channel]
        showing_images1 = tf.concat(showing_images, 1)
    
        showing_images = images[:grid_size*grid_size,:,:,3:6]
        showing_images = tf.reshape(showing_images, [-1, width, 3])
        showing_images = tf.split(showing_images, grid_size, 0)
        showing_images2 = tf.concat(showing_images, 1)
    
        showing_images = images[:grid_size*grid_size,:,:,6:9]
        showing_images = tf.reshape(showing_images, [-1, width, 3])
        showing_images = tf.split(showing_images, grid_size, 0)
        showing_images3 = tf.concat(showing_images, 1)

        return tf.expand_dims(showing_images1, 0), tf.expand_dims(showing_images2, 0), tf.expand_dims(showing_images3, 0)

    else:
        # To think easily, assume no channel
        showing_images = images[:grid_size*grid_size,:,:,:3]
        showing_images = tf.reshape(showing_images, [-1, width, channel])
        # tf.split, if num_or_size_split is an integer, split value along axis into num_split tensors
            # if num_or_size_split is not an integer, it is presumed as a split size
        showing_images = tf.split(showing_images, grid_size, 0)
        # [height*grid_size, width*grid_size, channel]
        showing_images = tf.concat(showing_images, 1)

        return tf.expand_dims(showing_images, 0)

def _merge_image_grid(source_images, transferred_images, max_grid_size=4, channels=3):

    if channels == 3:
        source_grid = _image_grid(source_images)
        transferred_grid = _image_grid(transferred_images)
        
        # check channel
        #assert source_grid.get_shape().as_list()[-1] == transferred_grid.get_shape().as_list()[-1]

        output_grid = tf.concat([source_grid, transferred_grid], 1)

        return output_grid

    else:
        source_grid1, source_grid2, source_grid3 = _image_grid(source_images)
        transferred_grid1, transferred_grid2, transferred_grid3 = _image_grid(transferred_images)
        
        # check channel
        #assert source_grid.get_shape().as_list()[-1] == transferred_grid.get_shape().as_list()[-1]
    
        output_grid1 = tf.concat([source_grid1, transferred_grid1], 1)
        output_grid2 = tf.concat([source_grid2, transferred_grid2], 1)
        output_grid3 = tf.concat([source_grid3, transferred_grid3], 1)
    
        return output_grid1, output_grid2, output_grid3


def _summarize_transferred_grid(source_images, transferred_images=None, name='Image_grid', channels=3):
    '''
        source_images, transferred_imags: [batch size, height, width, channels]
    '''
    if transferred_images is not None:
        transferred_channels = transferred_images.get_shape().as_list()[-1]
    
        if channels == 3:
            grid = _merge_image_grid(source_images, transferred_images, channels=channels)
            # max_outputs: max number of batch to generate images
            # name/Image
            return tf.summary.image('%s_image_grid' % name, grid, max_outputs=1)
        else:
            grid1, grid2, grid3 = _merge_image_grid(source_images, transferred_images, channels=channels)
            # max_outputs: max number of batch to generate images
            # name/Image
            return tf.summary.image('%s_image_grid' % name, grid1, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid2, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid3, max_outputs=1)

    else:
        if channels == 3:
            grid = _image_grid(source_images)
            return grid
        else:
            grid1, grid2, grid3 = _image_grid(source_images)
            return tf.summary.image('%s_image_grid' % name, grid1, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid2, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid3, max_outputs=1), 


def config_summary(save_dir, s2t_adversarial_weight, t2s_adversarial_weight, s2t_cyclic_weight, t2s_cyclic_weight, s2t_task_weight, t2s_task_weight, discriminator_step, generator_step, adversarial_mode, whether_noise, noise_dim, s2t_style_weight, t2s_style_weight):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        f.write(str(adversarial_mode) + '\n')
        f.write('\ns2t adversarial weight : ' + str(s2t_adversarial_weight))
        f.write('\nt2s adversarial weight : ' + str(t2s_adversarial_weight))
        f.write('\ns2t cyclic weight : ' + str(s2t_cyclic_weight))
        f.write('\nt2s cyclic weight : ' + str(t2s_cyclic_weight))
        f.write('\ns2t task weight : ' + str(s2t_task_weight))
        f.write('\nt2s task weight : ' + str(t2s_task_weight))
        f.write('\ns2t style weight : ' + str(s2t_style_weight))
        f.write('\nt2s style weight : ' + str(t2s_style_weight))
        f.write('\nDiscriminator step : ' + str(discriminator_step))
        f.write('\nGenerator step : ' + str(generator_step))
        f.write('\nNoise : ' + str(whether_noise))
        f.write('\nNoise dimension : ' + str(noise_dim))
        f.close()



