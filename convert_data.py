import tensorflow as tf
import os
import importlib
import numpy as np
import dataset_utils
import math
import utils
import op
slim = tf.contrib.slim
import sys
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import pickle
import shutil

def hist_matching(content, style):
    content = (content + 1) /2 * 255.
    style   = (style +1) /2 * 255.
    imres = np.ones(content.shape) * 255
    for d in range(3):
        stylehist, bins = np.histogram(style[:,:,:,d].flatten(), bins=255, range=(0,255), normed=True)
        stylecdf = stylehist.cumsum()
        stylecdf = (255 * stylecdf / stylecdf[-1]).astype(np.uint8)

        for i in range(content.shape[0]):
            contenthist, _ = np.histogram(content[i,:,:,d].flatten(), bins=255, range=(0,255),normed=True)
            contentcdf = contenthist.cumsum()
            contentcdf = (255 * contentcdf / contentcdf[-1]).astype(np.uint8)
            im2 = np.interp(content[i,:,:,d], bins[:-1],contentcdf)
            imres[i,:,:,d] = np.interp(im2,stylecdf, bins[:-1])

    imres = (imres - 127.5) /127.5
    return imres

def convert(sess, args, config):

    #from scipy.misc import imread, imsave
    #imsrc = imread('front_215.jpeg')
    #imst  = imread('left_92.jpeg')

    #imres = hist_matching(imsrc, imst)
    #imsave('matching.jpeg',imres) 

    model_type = config.get('config', 'experiment')
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    im_dir = os.path.join(base_dir,'transferred/imgs')
    data_dir = os.path.join(base_dir,'transferred')
    log_dir = os.path.join(base_dir, config.get('config', 'logdir'))
    tfrecord_dir = os.path.join(base_dir, config.get(model_type, 'tfrecord'))
    ckpt_dir = os.path.join(log_dir, utils.make_savedir(config))
    print(ckpt_dir)

    if os.path.exists(im_dir):
        shutil.rmtree(im_dir,ignore_errors=True)
    os.mkdir(im_dir)   
 
    model_file = importlib.import_module(model_type)
    model = getattr(model_file, 'model')

    da_model = model(args, config)

    get_batches = getattr(dataset_utils, model_type)
    
    # Get test batches   
    with tf.name_scope('batch'):
        source_image_batch, source_label_batch, source_measure_batch, source_command_batch = get_batches('source', 'train', tfrecord_dir, batch_size=args.batch_size, config=config, args=args)
        target_image_batch, target_label_batch, target_measure_batch, target_command_batch = get_batches('target', 'test', tfrecord_dir, batch_size=args.batch_size, config=config, args=args)
    # Call model
    da_model(source_image_batch, target_image_batch, source_measure_batch)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    # Load checkpoint 
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info('Checkpoint loaded from %s' % ckpt_dir)
    else:
        tf.logging.warn('Checkpoint not loaded')
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
      label_list = []
      measure_list = []
      command_list = []
      for i in range(1000):
        transferred_im, label, measure, command \
                 = sess.run([da_model.summary['source_transferred'],source_label_batch, source_measure_batch, source_command_batch])
        for j in range(args.batch_size):
            left_im = transferred_im[j,:,:,0:3]
            front_im = transferred_im[j,:,:,3:6]
            right_im = transferred_im[j,:,:,6:]
            left_filename = 'left_%d.jpeg' % (i*args.batch_size + j)
            front_filename = 'front_%d.jpeg' % (i*args.batch_size + j)
            right_filename = 'right_%d.jpeg' % (i*args.batch_size + j)
            imsave(os.path.join(im_dir,'left_%d.jpeg' % (i*args.batch_size + j)), left_im)
            imsave(os.path.join(im_dir,'front_%d.jpeg' % (i*args.batch_size + j)), front_im)
            imsave(os.path.join(im_dir,'right_%d.jpeg' % (i*args.batch_size + j)), right_im)
            label_list.append(label[j,:])
            measure_list.append(measure[j,:])
            command_list.append(command[j,:])
      
      with open(os.path.join(data_dir,'data0.pkl'),'wb') as f:
        pickle.dump([label_list,measure_list,command_list],f)    
                

    finally:
        coord.request_stop()
        coord.join(threads)

def accuracy(left_cmd, str_cmd, right_cmd, batch_size, command, num_labels=5, print_info=False):
    steer_result = list()
    steer = np.concatenate([left_cmd, str_cmd, right_cmd], axis=1)
    steer = steer.reshape([batch_size, command.shape[-1], num_labels])
    steer = steer * np.expand_dims(command, 2)
    for i in range(batch_size):
        steer_result.append(steer[i, command[i] != 0])

    steer_result = np.concatenate(steer_result, 0)

    if print_info:
        print('LEFT COMMAND')
        print(left_cmd)
        print('STRAIGHT COMMAND')
        print(str_cmd)
        print('RIGHT_COMMAND')
        print(right_cmd)

    print('RESULT')
    print(steer_result)       


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:."+ str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s | %s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush() 


