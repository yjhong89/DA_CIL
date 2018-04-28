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


def evaluation(sess, args, config):
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

    # Call model
    da_model(source_image_batch, None, source_measure_batch)

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
        for idx in range(args.num_eval):
            # tf.nn.softmax's default axis is -1 (last dimension)
            command_l = tf.nn.softmax(da_model.end[0])
            command_s = tf.nn.softmax(da_model.end[1])
            command_r = tf.nn.softmax(da_model.end[2])
            command, steer_label, np_im, steer_prediction_l, steer_prediction_s, steer_prediction_r = sess.run([source_command_batch, source_label_batch, source_image_batch, command_l, command_s, command_r])
            print('COMMAND LABEL')
            print(command)
            #steer_pred = np.concatenate(steer_prediction, axis=1)
            # Regression module has 3 branches for each command.
            print('STEER')
            print(steer_label)
            #print('-'*20)
            #print('LEFT COMMAND')
            #print(steer_prediction_l)
            #print('STRAIGHT COMMAND')
            #print(steer_prediction_s)
            #print('RIGHT_COMMAND')
            #print(steer_prediction_r)
            
            print('RESULT')
            steer = np.concatenate([steer_prediction_l, steer_prediction_s, steer_prediction_r], axis=1)
            steer = steer.reshape([args.batch_size, command.shape[-1], steer_label.shape[-1]])
            steer = steer * np.expand_dims(command, 2)
            steer_result = list()
            for i in range(args.batch_size):
                steer_result.append(steer[i, command[i]!=0])
                
            steer_result = np.concatenate(steer_result, 0)
            print(steer_result)
            print('='*20)
            

    finally:
        coord.request_stop()
        coord.join(threads)
       


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:."+ str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s | %s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush() 


