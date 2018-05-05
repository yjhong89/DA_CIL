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

def evaluation(sess, args, config):

    #from scipy.misc import imread, imsave
    #imsrc = imread('front_215.jpeg')
    #imst  = imread('left_92.jpeg')

    #imres = hist_matching(imsrc, imst)
    #imsave('matching.jpeg',imres) 

    model_type = config.get('config', 'experiment')
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    log_dir = os.path.join(base_dir, config.get('config', 'logdir'))
    tfrecord_dir = os.path.join(base_dir, config.get(model_type, 'tfrecord'))
    ckpt_dir = os.path.join(log_dir, utils.make_savedir(config))
    print(ckpt_dir)
    
    model_file = importlib.import_module(model_type)
    model = getattr(model_file, 'model')

    da_model = model(args, config)

    get_batches = getattr(dataset_utils, model_type)
    
    # Get test batches   
    with tf.name_scope('batch'):
        source_image_batch, source_label_batch, source_measure_batch, source_command_batch = get_batches('source', 'test', tfrecord_dir, batch_size=args.batch_size, config=config)
        target_image_batch, target_label_batch, target_measure_batch, target_command_batch = get_batches('target','test',tfrecord_dir, batch_size=args.batch_size, config=config)

    # Call model
    da_model(source_image_batch, target_image_batch, source_measure_batch)

    with tf.name_scope('objective'):
        da_model.create_objective(source_label_batch, source_command_batch)

    command_l = tf.nn.softmax(da_model.end[0])
    command_s = tf.nn.softmax(da_model.end[1])
    command_r = tf.nn.softmax(da_model.end[2])
    command_l_one_hot = slim.one_hot_encoding(tf.argmax(command_l, 1), args.num_label)
    command_s_one_hot = slim.one_hot_encoding(tf.argmax(command_s, 1), args.num_label)
    command_r_one_hot = slim.one_hot_encoding(tf.argmax(command_r, 1), args.num_label)

    labels = slim.one_hot_encoding(tf.argmax(target_label_batch, 1), args.num_label)
    
    command_l_labels = tf.reduce_sum(tf.expand_dims(target_command_batch[:,0], 1) * labels, 0)
    command_s_labels = tf.reduce_sum(tf.expand_dims(target_command_batch[:,1], 1) * labels, 0)
    command_r_labels = tf.reduce_sum(tf.expand_dims(target_command_batch[:,2], 1) * labels, 0)
    
    command_l_correct = tf.reduce_sum(tf.expand_dims(target_command_batch[:,0], 1) * tf.multiply(labels, command_l_one_hot), 0)
    command_s_correct = tf.reduce_sum(tf.expand_dims(target_command_batch[:,1], 1) * tf.multiply(labels, command_s_one_hot), 0)
    command_r_correct = tf.reduce_sum(tf.expand_dims(target_command_batch[:,2], 1) * tf.multiply(labels, command_r_one_hot), 0)   

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
        left_label_num = np.zeros(5)
        str_label_num  = np.zeros(5)
        right_label_num= np.zeros(5)
        left_acc       = np.zeros(5)
        str_acc        = np.zeros(5)
        right_acc      = np.zeros(5)   
        for idx in range(args.num_eval):
            sim, tim, s2tim, t2sim = sess.run([da_model.source_concat,da_model.target_concat, da_model.g_s2t, da_model.g_t2s])
            s2tim_mat = hist_matching(s2tim, tim)
            t2sim_mat = hist_matching(t2sim, sim)
            imsave('s2tim.png',s2tim[0,:,:,:])
            imsave('s2tim_mat.png',np.uint8((s2tim_mat[0,:,:,:]+1)/2*255))
            imsave('t2sim.png',t2sim[0,:,:,:])
            imsave('t2sim_mat.png',t2sim_mat[0,:,:,:])
            command, steer_label, np_im, steer_pred_l, steer_pred_s, steer_pred_r, cmd_l_label_num, cmd_s_label_num, cmd_r_label_num, cmd_l_correct, cmd_s_correct, cmd_r_correct \
                 = sess.run([target_command_batch, target_label_batch, target_image_batch, command_l, command_s, command_r, command_l_labels, command_s_labels, command_r_labels, command_l_correct, command_s_correct, command_r_correct])

            if args.print_info:
                print('COMMAND LABEL')
                print(command)
                print('STEER')
                print(steer_label)
                print('-'*20)
                accuracy(steer_pred_l, steer_pred_s, steer_pred_r, args.batch_size, command, args.num_label)
                print(cmd_l_label_num)
                print(cmd_s_label_num)
                print(cmd_r_label_num)
                print('-'*10)
                print(cmd_l_correct)
                print(cmd_s_correct)
                print(cmd_r_correct) 

            left_label_num += cmd_l_label_num
            str_label_num += cmd_s_label_num
            right_label_num += cmd_r_label_num
            left_acc += cmd_l_correct
            str_acc += cmd_s_correct
            right_acc += cmd_r_correct

        for i in range(args.num_label):
            left_acc[i] = left_acc[i] / left_label_num[i]
            str_acc[i] = str_acc[i] / str_label_num[i]
            right_acc[i] = right_acc[i] / right_label_num[i]

        print('LEFT COMMAND ACCURACY')
        print(left_acc)
        print('STRAIGHT COMMAND ACCURACY')
        print(str_acc)
        print('RIGHT COMMAND ACCURACY')
        print(right_acc)
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


