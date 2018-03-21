import tensorflow as tf
import h5py
import os
import glob
import numpy as np
import sys


def tfrecord_path(tfrecord_dir, whether_for_source, data_type): 
    if whether_for_source:
        return os.path.join(tfrecord_dir, 'source_' + data_type + '.tfrecord')
    else:
        return os.path.join(tfrecord_dir, 'target_' + data_type + '.tfrecord')

def process_h5file(data_path, writer):
    tf.logging.info('Loading %s' % data_path)
    file_list = glob.glob(os.path.join(data_path, '*.h5'))
    
    for f in file_list:
        try:
            tf.logging.info('Read %s' % f)
            hdf_content = h5py.File(f, 'r')
            imgs = hdf_content.get('rgb')
            lbls = hdf_content.get('targets')
            if len(imgs) != len(lbls):
                raise Exception
            num_files = len(imgs)

        except OSError as e:
            print('Error:', e)
            print(f)
            continue

        # num_files: 200
        for i in range(num_files):
            rgb_data = imgs[i].astype(np.float32)
            # Get image size: [height, width]: [88, 200]
            image_size = rgb_data.shape[:-1]
            # Get measures
            steer, acc, s, o, t, c = _get_stats(lbls[i])

            max_steer, max_acc, max_speed, max_or = _check_max(steer, acc, s, o)
        
            example = _make_tfexample(rgb_data, steer, acc, s, o, t, c)
            # Write
            writer.write(example.SerializeToString())

    tf.logging.info('%s files has been completed into tfrecord' % data_path)
    print('Max speed, ', max_speed)
    print('Max_steer, ', max_steer)
    print('Max_orientation, ', max_or)
    print('max_acceleration, ', max_acc) 
    sys.stdout.write('\n')
    # Flush the buffer, write everything in the buffer to the terminal
    sys.stdout.flush()

def _check_max(steer, acceleration, speed, orientation):
    max_steer = 0
    max_speed = 0
    # x,y,z
    max_acc = [0]*3
    max_orientation = [0]*3

    if max_steer < abs(steer):
        max_steer = abs(steer)
    if max_speed < abs(speed):
        max_speed = abs(speed)
    for n, i in enumerate(zip(max_acc, max_orientation)):
        if i[0] < abs(acceleration[n]):
            max_acc[n] = abs(acceleration[n])
        if i[1] < abs(orientation[n]):
            max_orientation[n] = abs(orientation[n])

    return max_steer, max_acc, max_speed, max_orientation


def _get_stats(label_info):
    # Already numpy file
    # Label: steering angle and acceleration
    steering_angle = label_info[0]
    acceleration = label_info[16:19]
    # Measure: speed, orientation, traffic_rule
    speed = label_info[10]
    # orientation_xyz
    orientation = label_info[21:24]
    # traffic_rule: opposite lane inter, sidewalk intersect
    traffic_rule = label_info[14:15]

    # command: (2, follow lane), (3, left), (4, right), (5, straight)
    command = int(label_info[24])
    command = _get_one_hot(command)

    return steering_angle, acceleration, speed, orientation, traffic_rule, command

def _get_one_hot(command):
    if not isinstance(command, int):
        command = int(command)

    one_hot = np.zeros([4,])
    one_hot[command-2] = 1

    return one_hot


def _make_tfexample(image, steering_angle, acceleration, speed, orientation, traffic_rule, command):
    return tf.train.Example(features=tf.train.Features(feature={
            'rgb_data': _bytes_features(tf.compat.as_bytes(image.tostring())),
            'label/steer': _float_features(steering_angle),
            'label/acc': _float_features(acceleration),
            'measures/speed': _float_features(speed),
            'measures/orientation': _float_features(orientation),
            'measures/traffic_rule': _float_features(traffic_rule), 
            'command': _bytes_features(tf.compat.as_bytes(command.tostring()))
            }))

def _bytes_features(value):
    if not isinstance(value, (tuple, list, np.ndarray)):
        value = [value]
    # value: string
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_features(value):
    # value: a integer scalar or list of integer scalar values
    if not isinstance(value, (tuple, list, np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    # value: a float scalar or list of float scalar values
    if not isinstance(value, (tuple, list, np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Read data from tfrecord file
def load_data(tfrecord_dir, whether_for_source, data_type, config):
    tfrecord_path = tfrecord_path(tfrecord_dir, whether_for_source, data_type)
    with tf.name_scope('read_tfrecord'):
        if not isinstance(tfrecord_path, (tuple, list)):
            tfrecord_path = [tfrecord_path]

        num_examples = sum(sum(1 for _ in tf.python_io.tf_Record_iterator(path)) for path in tfrecord_path)
        tf.logging.info('%d examples' % num_examples)

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

    labels = (normalized_steer, normalized_acc, normalized_speed)

    return image, labels, command



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


