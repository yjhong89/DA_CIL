import tensorflow as tf
import os
import argparse
import configparser
import utils
from train import train
#import inference

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--training', type=utils.str2bool, default='t')
    parser.add_argument('--max_iter', type=int, default=200000)
    parser.add_argument('--t2s_task', type=utils.str2bool, default='t')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=utils.str2bool, default='y')
    parser.add_argument('--input_mask', type=utils.str2bool, default='f')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='LS')   

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


if __name__ == "__main__":
    main()
