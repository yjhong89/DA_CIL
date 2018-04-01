import tensorflow as tf
import os
import importlib
import numpy as np

def evaluation(sess, args, config):
    model_type = config.get('config', 'experiment')
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    log_dir = os.path.join(base_dir, config.get('config', 'logdir'))
    tfrecord_dir = os.path.join(base_dir, config.get(model_type, 'tfrecord'))
    to_save_dir = config.get('config', 'savedir')
    ckpt_dir = os.path.join(log_dir, to_save_dir)

    saver = tf.train.Saver()

    model_file = importlib.import_module(model_type)
    model = getattr(model_file, 'model')

    da_model = model(args, config)
    
    # Get test batches   
    with tf.name_scope('batch'):
        source_image_batch, source_label_batch = dataset_utils.get_batches('source', 'test', tfrecord_dir, batch_size=args.batch_size, config=config)
        mask_image_batch = source_image_batch[:,:,:,3]
        source_image_batch = source_image_batch[:,:,:,:3]
        if args.input_mask:
            # 0/1 -> -1/1, -1 would be 0
            mask_images = tf.to_float(tf.greater(mask_image_batch, 0.99))
            source_image_batch = tf.multiply(source_image_batch, tf.tile(tf.expand_dims(mask_images, [1,1,1,3]))

        target_image_batch, _ = dataset_utils.get_batches('target', 'test', tfrecord_dir, batch_size=args.batch_size, config=config)

    # Call model
    da_model(source_image_batch, target_image_batch)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    # Load checkpoint 
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
        target_head_prob, target_lateral_prob = sess.run([tf.nn.softmax(da_model.target_head_logits), tf.nn.softmax(da_model.target_lateral_logits)]) 


    finally:
        coord.request_stop()
        coord.join(threads)
        
