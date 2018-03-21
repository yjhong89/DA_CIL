import tensorflow as tf
import numpy as np
import os
import configparser
import dataset_utils
import inspect

class MyConfigParser(configparser.ConfigParser):
    def getlist(self, section, option):
        value = self.get(section, option)
        return list(filter(None, value.split(',')))

    def getlistint(self, section, option):
        return [int(x) for x in self.getlist(section, option)]
    
def load_config(config, ini):
    config_file = os.path.expanduser(ini)
    config.read(config_file)


def str2bool(v):
    if v.lower() in ('true', 't', 'y', 'yes'):
        return True
    elif v.lower() in ('false', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError('%s is not supported' % v)

# Read data from tfrecord file
def load_data(tfrecord_dir, whether_for_source, data_type, config):
    tfrecord_path = dataset_utils.tfrecord_path(tfrecord_dir, whether_for_source, data_type)
    with tf.name_scope('read_tfrecord'):
        if not isinstance(tfrecord_path, (tuple, list)):
            tfrecord_path = [tfrecord_path]

        file_queue = tf.train.string_input_producer(tfrecord_path)
        reader = tf.TFRecordReader()
        _, serialzied_example = reader.read(file_queue)

        keys_to_features = {
            'rgb_data': tf.FixedLenFeature([], tf.string),
            'label/steer': tf.FixedLenFeature([1], tf.float32),
            'label/acc': tf.FixedLenFeature([3], tf.float32),
            'measures/speed': tf.FixedLenFeature([1], tf.float32),
            'measures/orientation': tf.FixedLenFeature([3], tf.float32),
            'measures/traffic_rule': tf.FixedLenFeature([2], tf.float32),
            'command': tf.FixedLenFeature([1], tf.int64)}

        example = tf.parse_single_example(serialized_example, features=keys_to_features)

    with tf.name_scope('decode'):
        # only tf.string need tf.decode_raw
        image = tf.decode_raw(example['rgb_data'], tf.float32, name='image')
        steer = tf.cast(example['label/steer'], tf.float32, name='steer')
        acc = tf.cast(example['label/acc'], tf.float32, name='acc')
        speed = tf.cast(example['measures/speed'], tf.float32, name='speed')
        orientation = tf.cast(example['measures/orientation'], tf.float32, name='orientation')
        traffic_rule = tf.cast(example['measures/traffic_rule'], tf.float32, name='traffic_rule')
        command = tf.decode_raw(example['command'], tf.int64, name='command')
       
    if config.getboolean('model', 'augmentation'):
        image = _augmentation(image, config)

    image = tf.clip_by_value(image, 0, 255.0)
    command = tf.one_hot

    # Normalize with the maximum value
    normalized_steer, normalized_acc, normalized_speed = tf.py_func(stats_normalize, [steer, acc, speed], [tf.float32]*3)

    with tf.name_scope('reshape_stats'):
        normalized_steer = tf.reshape(normalized_steer, [1])
        normalized_speed = tf.reshape(normalized_speed, [1])
        normalized_acc = tf.reshape(normalized_acc, [3])

    stats = (normalized_steer, normalized_acc, normalized_speed)

    return image, stats, command



def _augmentation(image, config):
    # inspect.stack get function name
    section = inspect.stack()[0][3]
    prob = config.getfloat(section, 'probability')
    with tf.name_scope(section):
        # Based on HSV, (Hue, Saturation, Value)
            # Hue: color, Saturation: intensity of color, Value: brightness of color
        # Saturation is the intensity of a color, determines the strength of a particular color
        # Make color look richer, more vivid
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_saturation(image, lower=0.5, upper=1.5),
                    lambda: image)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_brightness(image, max_delta=63),
                    lambda: image)
        # Hue is color
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_hue(image, max_delta=0.032),
                    lambda: image)
        # Contrast is the difference between amximum and minimum pixel intensity
            # Dark parts darker and the light parts lighter
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                    lambda: image)
        # Gaussian noise
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], 5, 15),
                    lambda: image)

    return image


