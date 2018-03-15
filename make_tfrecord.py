import tensorflow as tf
import argparse
import shutil
import configparser
import dataset_utils
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini')
    parser.add_argument('-t', '--data_type', nargs='+', default=['train', 'test'])
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('--source', type=dataset_utils.str2bool, default='t', help='Whether for source')

    args = parser.parse_args()
    if args.log:    
        tf.logging.set_verbosity('INFO')

    # Current directory
    base_dir = os.getcwd()
    # Read config file
    config = configparser.ConfigParser()
    dataset_utils.load_config(config, args.config)

    carla_data_dir = os.path.join(base_dir, config.get('directory', 'carla'))    
    tfrecord_dir = os.path.join(base_dir, config.get('directory', 'tfrecord'))    
    
    if args.delete:
        tf.logging.warn('Delete existing tfrecord files')
        shutil.rmtree(tfrecord_dir)

    os.makedirs(tfrecord_dir, exist_ok=True)   

    for t in args.data_type:
        tfrecord_path = dataset_utils.tfrecord_path(tfrecord_dir, args.source, t) 
        tf.logging.info('Writing %s.tfrecord file' % t)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            data_path = os.path.join(carla_data_dir, config.get('directory', t))
            if not os.path.exists(data_path):
                raise Exception('No %s directory' % data_path)
            
            dataset_utils.process_h5file(data_path, writer)



if __name__ == "__main__":
    main()
