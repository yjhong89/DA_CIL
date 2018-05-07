import tensorflow as tf
import h5py
import os
import glob
import inspect
import numpy as np
import sys
import tensorflow.contrib.slim as slim

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

regressionList = [-0.5,-0.3,-0.1,0.1,0.3,0.5]
 

def tfrecord_path(tfrecord_dir, dataset, data_type): 
    assert dataset in ['source','target','transferred']
    return os.path.join(tfrecord_dir, dataset +'_' + data_type + '.tfrecord')

def process_h5file(data_path, writer, data_type,dataset):
    tf.logging.info('Loading %s for %s' % (data_path, data_type))
    file_list = glob.glob(os.path.join(data_path, '*.hdf5'))
    
    for f in file_list:
        try:
            tf.logging.info('Read %s' % f)
            hdf_content = h5py.File(f, 'r')
            if data_type == 'train':
              imgs = hdf_content.get('tr_img')
              lbls = hdf_content.get('tr_labels')
              conds = hdf_content.get('tr_conds')
              measures = hdf_content.get('tr_measure')
            else:
              imgs = hdf_content.get('ts_img')
              lbls = hdf_content.get('ts_labels')
              conds = hdf_content.get('ts_conds')
              measures = hdf_content.get('ts_measure')

            if len(imgs) != len(lbls):
                raise Exception
            num_files = len(imgs)
            print(num_files)
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
            ang_z, linacc_x, linacc_y = _get_stats(measures[i])
            if dataset == 'transferred':
              lbls_class = lbls[i]
            else:
              for j in range(1,len(regressionList)):
                if lbls[i] <= regressionList[j]:
                  lbls_clss = j -1
                  break 
            example = _make_tfexample(rgb_data, _get_one_hot(lbls_class,soft_class=True,class_num=5), np.array([ang_z]), np.array([linacc_x]), np.array([linacc_y]), _get_one_hot(conds[i],soft_class=False,class_num=3))
            # Write
            writer.write(example.SerializeToString())

    tf.logging.info('%s files has been completed into tfrecord' % data_path)
    sys.stdout.write('\n')
    # Flush the buffer, write everything in the buffer to the terminal
    sys.stdout.flush()


def _get_stats(label_info):
    # Already numpy file
    # Label: steering angle and acceleration
    ang_z = label_info[0]
    linacc_x = label_info[1]
    linacc_y = label_info[2]
    # command: (2, follow lane), (3, left), (4, right), (5, straight)
    #command = int(label_info[24])
    #command = _get_one_hot(command)

    return ang_z, linacc_x, linacc_y

def _get_one_hot(command, soft_class=False, class_num=3):
    if not isinstance(command, int):
        command = int(command)

    one_hot = np.zeros([class_num,])
    if soft_class:
      if command == 0:
        one_hot[command] = 0.8
        one_hot[command+1] = 0.2
      elif command == class_num -1:
        one_hot[command] = 0.8
        one_hot[command-1] = 0.2
      else:
        one_hot[command] = 0.8
        one_hot[command-1] = one_hot[command+1] = 0.1
    else:
      one_hot[command] = 1

    return one_hot.astype(np.float32)


def _make_tfexample(image, steering, ang_z, linacc_x, linacc_y, command):
    return tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_features(tf.compat.as_bytes(image.tostring())),
            'label/steer': _bytes_features(tf.compat.as_bytes(steering.tostring())),
            'measures/ang_z': _float_features(ang_z),
            'measures/linacc_x': _float_features(linacc_x),
            'measures/linacc_y': _float_features(linacc_y),
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


# Image value: float type [0,1]
def _augmentation(image, config):
    # inspect.stack get function name
    section = inspect.stack()[0][3][1:]
    prob = config.getfloat(section, 'probability')
    with tf.name_scope(section):
        # Based on HSV, (Hue, Saturation, Value)
            # Hue: color, Saturation: intensity of color, Value: brightness of color
        # Saturation is the intensity of a color.
        # Make color look richer, more vivid
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_saturation(image, lower=0.8, upper=1.5),
                    lambda: image)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_brightness(image, max_delta=0.1),
                    lambda: image)
        # Hue is color
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_hue(image, max_delta=0.005),
                    lambda: image)
        # Contrast is the difference between amximum and minimum pixel intensity
            # Dark parts darker and the light parts lighter
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: tf.image.random_contrast(image, lower=0.8, upper=1.5),
                    lambda: image)
        # Gaussian noise
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([], maxval=1.0) < prob,
                    lambda: image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], 0, 0.01),
                    lambda: image)

    return image


def pixel_da(dataset_name, split_name, tfrecord_dir, batch_size, config=None):
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
        
        # Just convert image data type
        image = tf.image.convert_image_dtype(image, tf.float32)

        if config.getboolean('config', 'augmentation'):
            # Augmentation
            if dataset_name == 'source':
                image_ = image[:,:,:3]
                mask_ = tf.expand_dims(image[:,:,3], axis=2)
                image_ = _augmentation(image_, config)
                image = tf.concat([image_, mask_], axis=2)
            elif dataset_name == 'target':
                image = _augmentation(image, config)
                
        # Need to clip value
        image = tf.clip_by_value(image, 0, 1.0)
        image -= 0.5
        image *= 2
#        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
                capacity=batch_size*10, num_threads=10, min_after_dequeue=batch_size*2)

        # If increase image size, results in OOM
        image_batch = tf.image.resize_images(image_batch, [256, 256])
        
        # [batch, 1] -> [batch size, 1, 9]
        label_batch = slim.one_hot_encoding(label_batch, 9)
        label_batch = tf.reshape(label_batch, [-1, 9])
         
    return image_batch, label_batch


def da_cil(dataset_name, split_name, tfrecord_dir, batch_size, config=None, args=None):
    tfrecord_path = os.path.join(os.getcwd(), tfrecord_dir, '%s_%s.tfrecord' % (dataset_name, split_name))
    if not isinstance(tfrecord_path, (tuple, list)):
        tfrecord_path = [tfrecord_path]

    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in tfrecord_path)
    tf.logging.info('%s_%s.tfrecord: %d examples' % (dataset_name, split_name, num_examples))

    with tf.name_scope('read_tfrecord'):
        # Remove 'num_epoch'-> causes queues to incur Out of Range error after num_epoch
        filename_queue = tf.train.string_input_producer(tfrecord_path)
        reader = tf.TFRecordReader()
    
        _, serialized_example = reader.read(filename_queue)
    
        features = tf.parse_single_example(
          serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label/steer': tf.FixedLenFeature([], tf.string),
            #'label/steer' : tf.FixedLenFeature([], tf.float32),
            'measures/ang_z': tf.FixedLenFeature([], tf.float32),
            'measures/linacc_x': tf.FixedLenFeature([], tf.float32),
            'measures/linacc_y': tf.FixedLenFeature([], tf.float32),
            'command' : tf.FixedLenFeature([], tf.string)})
       
        image = tf.decode_raw(features['image_raw'],tf.float32)
        # steer is one hot string
        label = tf.decode_raw(features['label/steer'], tf.float32)
        #label = features['label/steer']
        angz = features['measures/ang_z']
        linx  = features['measures/linacc_x']
        liny  = features['measures/linacc_y']
        command = tf.decode_raw(features['command'],tf.float32)
    
        # tf.string must be notified its shape
        command = tf.reshape(command,[3])
        label = tf.reshape(label, [5])
        #image = tf.reshape(image,[240,360,3])
        image = tf.reshape(image, [240,360,9])
    
        image /= 255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
    
        if config.getboolean('config', 'augmentation'):
            image1 = _augmentation(image[:,:,:3], config)
            image2 = _augmentation(image[:,:,3:6], config)
            image3 = _augmentation(image[:,:,6:9], config)
    
        image = tf.concat([image1, image2, image3], axis=-1)
    
        image = tf.clip_by_value(image, 0, 1.0)
        image -= 0.5
        image *= 2
   
        if args.convert_data:
            images, labels, angzs, linxs, linys, commands = tf.train.shuffle_batch([image,label,angz,linx,liny,command],batch_size=batch_size, capacity=10*batch_size, num_threads=4, min_after_dequeue=batch_size)
        else: 
            images, labels, angzs, linxs, linys, commands = tf.train.batch([image,label,angz,linx,liny,command],batch_size=batch_size, capacity=10*batch_size, num_threads=4)
   
        generator_type = config.get('generator', 'type')
        if generator_type == 'UNET':
            images = tf.image.resize_images(images, [128, 128])
        else:
            images = tf.image.resize_images(images, [96, 96])
        
    
        angzs = tf.expand_dims(angzs, 1)
        linxs = tf.expand_dims(linxs, 1)
        linys = tf.expand_dims(linys, 1)
        measures = tf.concat([angzs,linxs,linys], 1)

    return images, labels, measures, commands


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

    config = utils.MyConfigParser()
    utils.load_config()

    with tf.Session() as sess:

        a, b, c, d = da_cil('source', 'train', 'pixel_da', 1, config)
        #check_tfrecord('target', 'train', 'pixel_da')
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(a))
        print(sess.run(c))

        coord.request_stop()
        coord.join(threads)
          
        
