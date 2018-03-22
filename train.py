import tensorflow as tf
import numpy as np
import os
import utils
import dataset_utils
import model
import shutil
import importlib

def _get_optimizer(config, name):
    section_name = 'optimizer'
    optimizer_dict = {
        'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate, config.getfloat(section, name+'_beta1'), config.getfloat(section, name+'_beta2'), config.getfloat(section, name+'_epsilon')),
        'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, config.getfloat(section, name+'_momentum')),
        'gd': lambda learning_rate: tf.train.GradientDescentOptimzer(learning_rate) }

    return optimizer_dict[name]

def _get_trainable_vars(scope):
    is_trainable = lambda x : x in tf.trainable_variables()

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    var_list = list(filter(is_trainable, var_list))

    tf.logging.info('Trainable variable for %s' % scope)
    tf.logging.info('%s' % var_list)

    return var_list

def _gradient_clip(name, optimizer, loss, clip_norm=5.0):
    var_list = _get_trainable_vars(name)
    grds, var = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    gradients = [None if grds is None else tf.clip_by_norm(gradient, 5.0) for gradient in grds]
    optim = optimizer.apply_gradients(zip(grads, var))
    return optim

def train(sess, args, config):
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    tfrecord_dir = os.path.join(base_dir, config.get('config', 'tfrecord'))
    log_dir = os.path.join(log_dir, config.get('config', 'logdir'))

    cyclic_hparam = config.getfloat('model', 'cyclic_hparam')
    regression_hparam = config.getfloat('model', 'regression_hparam')

    if args.delete:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_path = importlib.import_module(config.get('config', 'experiment'))
    model = getattr(model_path, 'model')

    da_model = model(args, config)

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.name_scope('batches'):
        source_image_batch, source_label_batch, source_command_batch = dataset_factory.get_batches(tfrecord_dir, whether_for_source=True, data_type=args.data_type, config=config)
        target_image_batch, _, _ = dataset_factpry.get_batches(tfrecord_dir, whether_for_source=False, data_type=args.data_type, config=config)

    da_model(source_image_batch, target_image_batch, source_label_batch[2])
   
    with tf.name_scope('objectives'):
        da_model.create_objective(source_label_batch[0], source_label_batch[1])
        generator_loss = da_model.g_step_loss()
        
        discriminator_loss = da_model.d_step_loss()   

    with tf.name_scope('optimizer'):
        if args.lr_decay:
            decay_steps = config.getint('optimizer', 'decay_steps')
            decau_rate = config.getfloat('optimizer', 'decay_rate')
            learning_rate = tf.train.exponential_decay(args.learning_rate, glbal_step, decay_steps, decay_rate, staircase=True)
        else:
            learning_rate = args.learning_rate

        optimizer = _get_optimizer(args.optimizer)(learning_rate)
        g_optim = _gradient_clip(name='generator', optimizer=optimizer, loss=generator_loss, clip_norm=args.clip_norm)
        d_optim = _gradient_clip(name='discriminator', optimizer=optimizer, loss=disicriminator_loss, clip_norm=args.clip_norm)
       
 
    generator_summary, discriminator_summary = utils.summarize(da_model.summary) 
    utils.config_summary(log_dir, config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for iter_count in range(args.max_iter):
            # Update discriminator
            for disc_iter in range(config.getint('model', 'discriminator_step')):
                loss, steps, disc_sum = sess.run([d_optim, global_step, discriminator_summary])
                writer.add_summary(disc_sum, steps)
                tf.logging.info('Step: %d: Discriminator loss=%.5f', steps, loss)

            for gen_iter in range(config.getint('model', 'generator_step')):
                loss, steps, gen_sum = sess.run([g_optim, global_step, generator_summary])
                writer.add_summary(gen_sum, steps)
                tf.logging.info('Step: %d: Generator loss=%.5f', steps, loss)
        
    except tf.errors.OutOfRangeError:
        print('Epoch limited')
    except KeyboardInterrupt:
        print('End training')
    finally:
        coord.request_stop()
        coord.join(thraeds=threads)      


