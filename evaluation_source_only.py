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
        for idx in range(10):
            command, steer_label, np_im, steer_prediction = sess.run([source_command_batch, source_label_batch, source_image_batch, da_model.end])

            steer_pred = np.concatenate(steer_prediction, axis=1)
            # Regression module has 3 branches for each command.
#            steer_prediction_left, steer_prediction_straight, steer_prediction_right = sess.run(da_model.end)
 #           steer_feature_left, steer_feature_straight, steer_feature_right = sess.run(da_model.end_feature)
            print('Steer label')
            print(steer_label)
            print('-'*20)
            print('Steer prediction with command')
            print(steer_pred * command)
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


