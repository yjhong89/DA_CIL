import tensorflow as tf
import h5py
import os
import glob
import numpy as np
import sys
import tensorflow.contrib.slim as slim

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

        num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in tfrecord_path)
        tf.logging.info(' %d examples' % num_examples)

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

_SOURCE_SPLITS_TO_SIZES = {'train': 67644, 'valid': 747, 'test': 6759}
_SOURCE_IMAGE_SIZE = [360,640,4]
_SOURCE_ITEMS_TO_DESCRIPTIONS = {
            'image': 'A [360x640x1] RGB image.',
            'label': 'A single integer between 0 and 8' }
_TARGET_SPLITS_TO_SIZES = {'train': 21870, 'valid': 2187, 'test': 2430}
_TARGET_IMAGE_SIZE = [180, 320, 3]
_TARGET_ITEMS_TO_DESCRIPTIONS = {
            'image': 'A [180x320x3]',
            'label': 'A single integer between 0 and 8'}

def get_batches(dataset_name, split_name, tfrecord_dir, batch_size):
    tfrecord_path = os.path.join(os.getcwd(), tfrecord_dir, '%s_%s.tfrecord' % (dataset_name, split_name))
    if not isinstance(tfrecord_path, (tuple, list)):
        tfrecord_path = [tfrecord_path]

    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in tfrecord_path)
    tf.logging.info('%s_%s.tfrecord: %d examples' % (dataset_name, split_name, num_examples))

    with tf.name_scope('read_tfrecord'):
        reader = tf.TFRecordReader
        #file_queue = tf.train.string_input_producer(tfrecord_path)
        #_, serialized_example = reader.read(file_queue)

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([1], tf.int64) } 

        if dataset_name == 'source':
            items_to_handlers = {
                'image': slim.tfexample_decoder.Image(shape=_SOURCE_IMAGE_SIZE, channels=4),
                'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]) }

            decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

            slim_dataset = slim.dataset.Dataset(data_sources=tfrecord_path, reader=reader, decoder=decoder, num_samples=_SOURCE_SPLITS_TO_SIZES[split_name], num_classes=9, items_to_descriptions=_SOURCE_ITEMS_TO_DESCRIPTIONS)

        elif dataset_name == 'target':
            items_to_handlers = {
                'image': slim.tfexample_decoder.Image(shape=_TARGET_IMAGE_SIZE, channels=3),
                'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]) }

            decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

            slim_dataset = slim.dataset.Dataset(data_sources=tfrecord_path, reader=reader, decoder=decoder, num_samples=_TARGET_SPLITS_TO_SIZES[split_name], num_classes=9, items_to_descriptions=_TARGET_ITEMS_TO_DESCRIPTIONS)
        else:
            raise ValueError
        
        provider = slim.dataset_data_provider.DatasetDataProvider(slim_dataset, num_readers=4, common_queue_capacity=20*batch_size, common_queue_min=10*batch_size)


        # currently, image dtype: uint8
        [image, label] = provider.get(['image', 'label'])
        # Images that are represented using floating point values are expected to have values in the range [0,1)
        # convert_image_dtype converts data type and scaling values
        image = tf.image.convert_image_dtype(image, tf.float32)
        #image = tf.image.per_image_standardization(image)
        image -= 0.5
        image *= 2

        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
                capacity=batch_size*10, num_threads=10, min_after_dequeue=batch_size*2)

        # If increase image size, results in OOM
        image_batch = tf.image.resize_images(image_batch, [90, 160])
        
        # [batch, 1] -> [batch size, 1, 9]
        label_batch = slim.one_hot_encoding(label_batch, 9)
        label_batch = tf.reshape(label_batch, [-1, 9])
         
        
#        example = tf.parse_single_example(serialized_example, features=keys_to_features)
#
#
#    with tf.name_scope('decode'):
#        image = tf.decode_raw(example['image/encoded'], tf.float32)
#        label = tf.cast(example['image/class/label'], tf.int64)
#
#        # Reshape image to original shape
#        # source image include mask in channel 4
#        if dataset_name == 'source':
#            image = tf.reshape(image, (360, 640, 4)) 
#        elif dataset_name == 'target':
#            image = tf.reshape(image, (90, 160, 3))
#
#        # per_sample_normalization
#        image = tf.image.per_image_standardization(image)
#        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
#                capacity=batch_size*5, num_threads=10, min_after_dequeue=10)
#
#        image_batch = tf.image.resize_images(image_batch, [360, 640])
#        
#        # [batch, 1] -> [batch size, 1, 9]
#        label_batch = slim.one_hot_encoding(label_batch, 9)
#        label_batch = tf.reshape(label_batch, [-1, 9])

    return image_batch, label_batch

def check_tfrecord(dataset_name, split_name, tfrecord_dir): 
    tfrecord_path = os.path.join(os.getcwd(), tfrecord_dir, '%s_%s.tfrecord' % (dataset_name, split_name))

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
    
    for string_record in record_iterator:
        # Parse next example
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        # Get features you stored
        label = int(example.features.feature['image/class/label']
                                .int64_list
                                .value[0])
        format_ = (example.features.feature['image/format']
                                .bytes_list
                                .value[0])

        img_string = (example.features.feature['image/encoded']
                                .bytes_list
                                .value[0])
        height = int(example.features.feature['image/height']
                                .int64_list
                                .value[0])
        width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])

        print('height/width', height, width)
        print(len(img_string),', ', label)
        # Convert image string to numpy, unsigned integer
        img = np.fromstring(img_string, np.uint8).astype(np.float32)



if __name__ == "__main__":
    c = tf.get_variable('a', [3,4]) 

    with tf.Session() as sess:

        a, b = get_batches('source', 'train', 'pixel_da', 10)
        #check_tfrecord('target', 'train', 'pixel_da')
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(a))
        print(sess.run(a).shape)
        print(sess.run(b))

        coord.request_stop()
        coord.join(threads)
          
        
