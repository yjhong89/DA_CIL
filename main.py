import tensorflow as tf
import os
import argparse
import configparser
import utils
import train
import inference

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = 2


def main():
    parser = argparser.ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_ture')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--gradient_norm', type=float, default=5.0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--train', type=utils.str2bool, default='t')

    args = parser.parse_args()
    if args.log:
        tf.logging.set_verbosity('INFO')

    config = utils.MyConfigParser()
    utils.load_config(config, args.config) 

    if args.train:
        tf.logging.info('Train')
    else:
        tf.logging.info('Inference')


if __name__ == "__main__":
    main()
