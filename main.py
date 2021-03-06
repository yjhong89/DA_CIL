import tensorflow as tf
import os
import argparse
import configparser
import utils
from train import train
import evaluation_pixel_da
import evaluation_source_only
import evaluation_da_cil

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--training', type=utils.str2bool, default='t')
    parser.add_argument('--max_iter', type=int, default=400000)
    parser.add_argument('--pixel_norm', type=utils.str2bool, default='t')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=utils.str2bool, default='n')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--summary_interval', type=int, default=50)
    parser.add_argument('--load_ckpt', type=utils.str2bool, default='t')
    parser.add_argument('--num_eval', type=int, default=100)
    parser.add_argument('--num_label', type=int, default=5)
    parser.add_argument('--print_info', type=utils.str2bool, default='n')
    parser.add_argument('--convert_data', type=utils.str2bool, default='t')

    args = parser.parse_args()
    if args.log:
        tf.logging.set_verbosity('INFO')

    config = utils.MyConfigParser()
    utils.load_config(config, args.config) 

    gpu_config = tf.ConfigProto()
    gpu_config.log_device_placement = False
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        if args.training:
            train(sess, args, config)
        else:
            tf.logging.info('Inference')
            if config.getboolean('config', 'source_only'):
                evaluation_source_only.evaluation(sess, args, config)
            else:
                experiment = config.get('config', 'experiment')
                if experiment == 'da_cil':
                    evaluation_da_cil.evaluation(sess, args, config)
                elif experiment == 'pixel_da':
                    evaluation_pixel_da.evaluation(sess, args, config)
                else:
                    raise ValueError('Not supported: %s' % experiment)


if __name__ == "__main__":
    main()
