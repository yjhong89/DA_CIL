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

    adversarial_weight = config.getint(model_type, 'adversarial_weight')
    cyclic_weight = config.getint(model_type, 'cyclic_weight')
    task_weight = config.getint(model_type, 'task_weight')

    result = model_type + '_' + adversarial_mode

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

def summarize(summary_set, t2s_option):
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
    
    assert channel == 3
    assert batch_size >= grid_size * grid_size

    # To think easily, assume no channel
    showing_images = images[:grid_size*grid_size,:,:,:]
    showing_images = tf.reshape(showing_images, [-1, width, channel])
    # tf.split, if num_or_size_split is an integer, split value along axis into num_split tensors
        # if num_or_size_split is not an integer, it is presumed as a split size
    showing_images = tf.split(showing_images, grid_size, 0)
    # [height*grid_size, width*grid_size, channel]
    showing_images = tf.concat(showing_images, 1)

    # tf.summary.image: image must be 4D
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
    transferred_channels = transferred_images.get_shape().as_list()[-1]

    if source_channels != transferred_channels:
        raise ValueError('Number of channels not match')

    if transferred_images is not None:
        grid = _merge_image_grid(source_images, transferred_images)
    else:
        grid = _image_grid(source_images)

    # max_outputs: max number of batch to generate images
    # name/Image
    return tf.summary.image('%s_image_grid' % name, grid, max_outputs=1)

def config_summary(save_dir, adversarial_weight, cyclic_weight, task_weight, discriminator_step, generator_step, adversarial_mode, whether_noise, noise_dim):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        f.write(str(adversarial_mode) + '\n')
        f.write('\nAdversarial weight : ' + str(adversarial_weight))
        f.write('\nCyclic weight : ' + str(cyclic_weight))
        f.write('\nTask weight : ' + str(task_weight))
        f.write('\nDiscriminator step : ' + str(discriminator_step))
        f.write('\nGenerator step : ' + str(generator_step))
        f.write('\nNoise : ' + str(whether_noise))
        f.write('\nNoise dimension : ' + str(noise_dim))
        f.close()



