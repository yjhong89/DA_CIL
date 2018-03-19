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
    # Make 0,1,2,3
    command = label_info[24] - 2

    return steering_angle, acceleration, speed, orientation, traffic_rule, command

def _make_tfexample(image, steering_angle, acceleration, speed, orientation, traffic_rule, command):
    return tf.train.Example(features=tf.train.Features(feature={
            'rgb_data': _bytes_features(tf.compat.as_bytes(image.tostring())),
            'label/steer': _float_features(steering_angle),
            'label/acc': _float_features(acceleration),
            'measures/speed': _float_features(speed),
            'measures/orientation': _float_features(orientation),
            'measures/traffic_rule': _float_features(traffic_rule), 
            'command': _int64_features(int(command))
            }))

def _bytes_features(value):
    # value: string
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

