import tensorflow as tf
import numpy as np
import os
import configparser
import dataset_utils
import inspect
import math

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
   # augmentation = config.getboolean('config', 'augmentation')
    noise = config.getboolean('generator', 'noise')
    mask = config.getboolean('config', 'input_mask')
    gen_type = config.get('generator', 'type')

    result = model_type + '_' + adversarial_mode + '_' + gen_type

    if noise:
        if mask:
            result = result + '_noise_mask'
        else:
            result = result + '_noise_wo_mask'
    else:
        if mask:
            result = result + '_wo_noise_mask'
        else:
            result = result + '_wo_noise_wo_mask'

    return result

def str2bool(v):
    if v.lower() in ('true', 't', 'y', 'yes'):
        return True
    elif v.lower() in ('false', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError('%s is not supported' % v)

def summarize(summary_set, t2s_option, source_only=False):
    if source_only:
        task_loss_summary = tf.summary.scalar('task_loss', summary_set['classification_loss'])
        source_image1_summary, source_image2_summary, source_image3_summary = _summarize_transferred_grid(summary_set['source_image'])
        return tf.summary.merge([task_loss_summary, source_image1_summary, source_image2_summary, source_image3_summary])

    # Loss part
    cyclic_summary = tf.summary.scalar('cyclic_loss', summary_set['cyclic_loss'])
    s2t_g_loss_summary = tf.summary.scalar('s2t_g_loss', summary_set['s2t_g_loss'])
    t2s_g_loss_summary = tf.summary.scalar('t2s_g_loss', summary_set['t2s_g_loss'])
    s2t_d_loss_summary = tf.summary.scalar('s2t_d_loss', summary_set['s2t_d_loss'])
    t2s_d_loss_summary = tf.summary.scalar('t2s_d_loss', summary_set['t2s_d_loss'])
    task_loss_summary = tf.summary.scalar('task_loss', summary_set['task_loss'])
    if t2s_option:
        t2s_task_loss_summary = tf.summary.scalar('t2s_task_loss', summary_set['t2s_task_loss'])
    else:
        t2s_task_loss_summary = tf.summary.scalar('t2s_regression_loss', 0)
    generator_loss_summary = tf.summary.scalar('generator_loss', summary_set['generator_loss'])
    discriminator_loss_summary = tf.summary.scalar('discriminator_loss', summary_set['discriminator_loss'])

    # Image part
    s2t_summary = _summarize_transferred_grid(summary_set['source_image'], summary_set['source_transferred'], name='S2T')
    t2s_summary = _summarize_transferred_grid(summary_set['target_image'], summary_set['target_transferred'], name='T2S')
    s2t2s_summary = _summarize_transferred_grid(summary_set['source_image'], summary_set['back2source'], name='S2T2S')
    t2s2t_summary = _summarize_transferred_grid(summary_set['target_image'], summary_set['back2target'], name='T2S2T')

    discriminator_merged = [s2t_d_loss_summary, t2s_d_loss_summary, task_loss_summary, t2s_task_loss_summary, discriminator_loss_summary]
    generator_merged = [cyclic_summary, s2t_g_loss_summary, t2s_g_loss_summary, generator_loss_summary, s2t_summary, t2s_summary, s2t2s_summary, t2s2t_summary]

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

def _merge_image_grid(source_images, transferred_images, max_grid_size=4):
    source_grid = _image_grid(source_images)
    transferred_grid = _image_grid(transferred_images)
    
    # check channel
    assert source_grid.get_shape().as_list()[-1] == transferred_grid.get_shape().as_list()[-1]

    output_grid = tf.concat([source_grid, transferred_grid], 1)

    return output_grid


def _summarize_transferred_grid(source_images, transferred_images=None, name='Image_grid'):
    '''
        source_images, transferred_imags: [batch size, height, width, channels]
    '''
    source_channels = source_images.get_shape().as_list()[-1]
    if transferred_images is not None:
        transferred_channels = transferred_images.get_shape().as_list()[-1]

        grid = _merge_image_grid(source_images, transferred_images)
        if source_channels != transferred_channels:
            raise ValueError('Number of channels not match')

    else:
        if source_channels == 3:
            grid = _image_grid(source_images)
        else:
            grid1, grid2, grid3 = _image_grid(source_images)
            return tf.summary.image('%s_image_grid' % name, grid1, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid2, max_outputs=1), tf.summary.image('%s_image_grid' % name, grid3, max_outputs=1), 

    # max_outputs: max number of batch to generate images
    # name/Image
    return tf.summary.image('%s_image_grid' % name, grid, max_outputs=1)

def config_summary(save_dir, s2t_adversarial_weight, t2s_adversarial_weight, s2t_cyclic_weight, t2s_cyclic_weight, s2t_task_weight, t2s_task_weight, discriminator_step, generator_step, adversarial_mode, whether_noise, noise_dim):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        f.write(str(adversarial_mode) + '\n')
        f.write('\ns2t_adversarial weight : ' + str(s2t_adversarial_weight))
        f.write('\nt2s_adversarial weight : ' + str(t2s_adversarial_weight))
        f.write('\ns2t_cyclic weight : ' + str(s2t_cyclic_weight))
        f.write('\nt2s_cyclic_weight : ' + str(t2s_cyclic_weight))
        f.write('\ns2t_task weight : ' + str(s2t_task_weight))
        f.write('\nt2s_task weight : ' + str(t2s_task_weight))
        f.write('\nDiscriminator step : ' + str(discriminator_step))
        f.write('\nGenerator step : ' + str(generator_step))
        f.write('\nNoise : ' + str(whether_noise))
        f.write('\nNoise dimension : ' + str(noise_dim))
        f.close()



