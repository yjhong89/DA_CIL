import tensorflow as tf
import argparse
import shutil
import configparser
import dataset_utils
import utils
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini')
    parser.add_argument('-t', '--data_type', nargs='+', default=['train', 'test'])
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('--dataset', default='source', help='select among source, target, transferred')

    args = parser.parse_args()
    if args.log:    
        tf.logging.set_verbosity('INFO')

    # Current directory
    base_dir = os.getcwd()
    # Read config file
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)

    if args.dataset=='source':
      data_mode = 'simul'
    elif args.dataset=='transferred':
      data_mode = 'transferred'
    else:
      data_mode = 'real'
    data_dir = os.path.join(base_dir, data_mode)
    #carla_data_dir = os.path.join(base_dir, config.get('config', 'carla'))    
    tfrecord_dir = os.path.join(base_dir, config.get('tfrecord', 'tfrecord'))    
    
    if args.delete:
        tf.logging.warn('Delete existing tfrecord files')
        shutil.rmtree(tfrecord_dir)

    if not os.path.exists(tfrecord_dir):
      os.makedirs(tfrecord_dir)   

    for t in args.data_type:
        tfrecord_path = dataset_utils.tfrecord_path(tfrecord_dir, args.dataset, t) 
        tf.logging.info('Writing %s.tfrecord file' % t)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            #data_path = os.path.join(carla_data_dir, config.get('config', t))
            if not os.path.exists(data_dir):
                raise Exception('No %s directory' % data_dir)
            
            dataset_utils.process_h5file(data_dir, writer,t,args.dataset)



if __name__ == "__main__":
    main()
