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
    
    # Get test batches   
    with tf.name_scope('batch'):
        source_image_batch, source_label_batch = dataset_utils.get_batches('source', 'test', tfrecord_dir, batch_size=args.batch_size, config=config)
        mask_image_batch = source_image_batch[:,:,:,3]
        source_image_batch = source_image_batch[:,:,:,:3]
        if config.getboolean('config','input_mask'):
            # 0/1 -> -1/1, -1 would be 0
            mask_images = tf.to_float(tf.greater(mask_image_batch, 0.99))
            source_image_batch = tf.multiply(source_image_batch, tf.tile(tf.expand_dims(mask_images,3), [1,1,1,3]))

        target_image_batch, target_label_batch = dataset_utils.get_batches('target', 'test', tfrecord_dir, batch_size=args.batch_size, config=config)
        target_label_max_batch = tf.argmax(target_label_batch, 1)
        target_lateral_label_batch = (target_label_max_batch % 9) / 3
        target_head_label_batch    = target_label_max_batch % 3

    # Call model
    da_model(source_image_batch, target_image_batch)

    #sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    # Load checkpoint 
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info('Checkpoint loaded from %s' % ckpt_dir)
    else:
        tf.logging.warn('Checkpoint not loaded')
        return

    target_lateral_one_hot_label = one_hot(target_lateral_label_batch)
    target_head_one_hot_label    = one_hot(target_head_label_batch)
    lateral_label_num     = tf.reduce_sum(target_lateral_one_hot_label, 0)
    head_label_num        = tf.reduce_sum(target_head_one_hot_label, 0)

    lateral_command       = get_command(da_model.target_lateral_logits)
    lateral_cmd_label     = tf.multiply(target_lateral_one_hot_label, lateral_command)
    head_command          = get_command(da_model.target_head_logits)
    head_cmd_label        = tf.multiply(target_head_one_hot_label, head_command)

    model_lateral_one_hot = one_hot(tf.argmax(da_model.target_lateral_logits,1))
    model_head_one_hot    = one_hot(tf.argmax(da_model.target_head_logits,1))

    lateral_correct_one   = tf.reduce_sum(tf.multiply(model_lateral_one_hot,target_lateral_one_hot_label),0)
    head_correct_one      = tf.reduce_sum(tf.multiply(model_head_one_hot,target_head_one_hot_label),0)
    
    target_lateral_one_hot_label = tf.expand_dims(target_lateral_one_hot_label,2)
    target_head_one_hot_label    = tf.expand_dims(target_head_one_hot_label,2)

    lateral_confusion_table = tf.multiply(target_lateral_one_hot_label, tf.expand_dims(tf.nn.softmax(da_model.target_lateral_logits),1))
    head_confusion_table = tf.multiply(target_head_one_hot_label, tf.expand_dims(tf.nn.softmax(da_model.target_head_logits),1))

    target_high_pass = op.neg_gaussian_filter(target_image_batch)
    trans_high_pass  = op.neg_gaussian_filter(da_model.g_s2t)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        #target_head_prob, target_lateral_prob = sess.run([tf.nn.softmax(da_model.target_head_logits), tf.nn.softmax(da_model.target_lateral_logits)]) 
        np_lateral_confusion = np.zeros((3,3))
        np_head_confusion    = np.zeros((3,3))
        np_lateral_num       = np.zeros(3)
        np_head_num          = np.zeros(3)
        np_lateral_acc       = np.zeros(3)
        np_head_acc          = np.zeros(3)
        np_lateral_cmd       = np.zeros(3)
        np_head_cmd          = np.zeros(3)
        np_l_conf_array      = np.zeros((1,3,3))
        np_h_conf_array      = np.zeros((1,3,3))
    
        test_batch_num = int(math.floor(20000 / args.batch_size))
    
        for idx in range(test_batch_num):
            if idx == 0:
                source, transferred_im, np_tar_high_pass, np_trans_high_pass = sess.run([source_image_batch, da_model.g_s2t, target_high_pass, trans_high_pass])
                import matplotlib.pyplot as plt
                import scipy.misc
                im_dir = os.path.join(ckpt_dir,'im')
                if not os.path.exists(im_dir):
                    os.mkdir(im_dir)
                for im_idx  in range(4):
                    scipy.misc.imsave(os.path.join(im_dir,'source%d.png' % im_idx), source[im_idx,:,:,:])
                    scipy.misc.imsave(os.path.join(im_dir,'trans%d.png' % im_idx), transferred_im[im_idx,:,:,:])
                    scipy.misc.imsave(os.path.join(im_dir,'trans_high%d.png' % im_idx), np.squeeze(np_trans_high_pass[im_idx,:,:,:]))
                    scipy.misc.imsave(os.path.join(im_dir,'tar_high%d.png' % im_idx), np.squeeze(np_tar_high_pass[im_idx]))

            t1,t2,x1,x2,y1,y2,z1,z2,a1,a2 = sess.run([lateral_confusion_table, head_confusion_table, tf.reduce_sum(lateral_confusion_table,0),tf.reduce_sum(head_confusion_table,0),
                                                        lateral_label_num, head_label_num,
                                                        lateral_correct_one, head_correct_one, tf.reduce_sum(lateral_cmd_label, 0), tf.reduce_sum(head_cmd_label,0)])
            np_lateral_confusion = np_lateral_confusion + x1
            np_head_confusion    = np_head_confusion + x2
            np_lateral_num       = np_lateral_num + y1
            np_head_num          = np_head_num + y2
            np_lateral_acc = np_lateral_acc + z1
            np_head_acc    = np_head_acc    + z2
            np_lateral_cmd += a1
            np_head_cmd    += a2
            np_l_conf_array = np.concatenate([np_l_conf_array,t1],0)
            np_h_conf_array = np.concatenate([np_h_conf_array,t2],0)
            printProgress(idx, test_batch_num, 'Progress', 'Complete', 1, 50)
        for i in range(3):
            np_lateral_confusion[i,:] /= np_lateral_num[i]
            np_head_confusion[i,:]    /= np_head_num[i]
            np_lateral_acc[i]         /= np_lateral_num[i]
            np_head_acc[i]            /= np_head_num[i]
            np_lateral_cmd[i]         /= np_lateral_num[i]
            np_head_cmd[i]            /= np_head_num[i]
        print('lateral_confusion')
        print(np_lateral_confusion)
        print('head_confusion')
        print(np_head_confusion)
        print('lateral-accuracy')
        print(np_lateral_acc)
        print('head-accuracy')
        print(np_head_acc)
        print('lateral cmd')
        print(np_lateral_cmd)
        print('head cmd')
        print(np_head_cmd)
    
        for i in range(3):
            y = np_l_conf_array[:,i,2] - np_l_conf_array[:,i,0]
            z = np_l_conf_array[:,i,0]
            print('when GT lateral is %d command mean : %.4f std : %.4f' % (i,np.mean(y[z!=0]),np.std(y[z!=0])))
            y = np_h_conf_array[:,i,2] - np_h_conf_array[:,i,0]
            z = np_h_conf_array[:,i,0]
            print('when GT head is %d command mean : %.4f std : %.4f' % (i,np.mean(y[z!=0]),np.std(y[z!=0])))
            
    


    finally:
        coord.request_stop()
        coord.join(threads)
       

def get_command(logit):
    return tf.tile(tf.expand_dims(tf.nn.softmax(logit)[:,2] - tf.nn.softmax(logit)[:,0],1), [1,3])

def one_hot(label):
    return slim.one_hot_encoding(tf.cast(label, tf.int64),3)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:."+ str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s | %s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush() 


