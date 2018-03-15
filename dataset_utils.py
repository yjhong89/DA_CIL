import tensorflow as tf
import h5py
import os
import glob
import numpy as np
import sys

def load_config(config, ini):
    config_file = os.path.expanduser(ini)
    config.read(config_file)

def tfrecord_path(tfrecord_dir, whether_for_source, data_type): 
    if whether_for_source:
        return os.path.join(tfrecord_dir, 'source_' + data_type + '.tfrecord')
    else:
        return os.path.join(tfrecord_dir, 'target_' + data_type + '.tfrecord')

def str2bool(v):
    if v.lower() in ('true', 't', 'y', 'yes'):
        return True
    elif v.lower() in ('false', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError('%s is not supported' % v)

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
            s, o, t, c = _get_measure_command(lbls[i])
        
            example = _make_tfexample(rgb_data, s, o, t, c)
            # Write
            writer.write(example.SerializeToString())

    tf.logging.info('%s files has been completed into tfrecord' % data_path)
    sys.stdout.write('\n')
    # Flush the buffer, write everything in the buffer to the terminal
    sys.stdout.flush()


def _get_measure_command(label_info):
    # Already numpy file
    # Measure: speed, orientation, traffic_rule
    speed = label_info[10]
    # orientation_xyz
    orientation = label_info[21:24]
    # traffic_rule: opposite lane inter, sidewalk intersect
    traffic_rule = label_info[14:15]

    # command: (2, follow lane), (3, left), (4, right), (5, straight)
    # Make 0,1,2,3
    command = label_info[24] - 2

    return speed, orientation, traffic_rule, command

def _make_tfexample(image, speed, orientation, traffic_rule, command):
    return tf.train.Example(features=tf.train.Features(feature={
            'rgb_data': _bytes_features(tf.compat.as_bytes(image.tostring())),
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

