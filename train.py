import tensorflow as tf
import numpy as np
import os
import utils
import dataset_utils
import da_cil
import pixel_da
import shutil
import importlib

def _get_optimizer(config, name):
    section = 'optimizer'
    optimizer_dict = {
        'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate, config.getfloat(section, name+'_beta1'), config.getfloat(section, name+'_beta2'), config.getfloat(section, name+'_epsilon')),
        'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, config.getfloat(section, 'momentum')),
        'gd': lambda learning_rate: tf.train.GradientDescentOptimzer(learning_rate),
        'rms': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate)         
}

    return optimizer_dict[name]

def _get_trainable_vars(scope):
    is_trainable = lambda x : x in tf.trainable_variables()

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    var_list = list(filter(is_trainable, var_list))

    tf.logging.info('Getting trainable variable for %s' % scope)
    #for i in var_list:
    #    print(i.op.name)
    #tf.logging.info('%s' % var_list)

    return var_list

def _gradient_clip(name, optimizer, loss, global_steps=None, clip_norm=5.0):
    var_list = _get_trainable_vars(name)
    grds, var = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    gradients = [gradient if gradient is None else tf.clip_by_value(gradient, -clip_norm, clip_norm) for gradient in grds]
    if global_steps is not None:
        optim = optimizer.apply_gradients(zip(gradients, var), global_step=global_steps)
    else:
        optim = optimizer.apply_gradients(zip(gradients, var))
    return optim

def train(sess, args, config):
    model_type = config.get('config', 'experiment')
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    tfrecord_dir = os.path.join(base_dir, config.get(model_type, 'tfrecord'))
    log_dir = os.path.join(base_dir, config.get('config', 'logdir'))
    adversarial_mode = config.get('config', 'mode')
    whether_noise = config.getboolean('generator', 'noise')
    t2s_task = config.getboolean('config', 't2s_task')
    noise_dim = config.getint('generator', 'noise_dim')

    source_only = config.getboolean('config', 'source_only')
    s2t_adversarial_weight = config.getfloat(model_type, 's2t_adversarial_weight')
    t2s_adversarial_weight = config.getfloat(model_type, 't2s_adversarial_weight')
    s2t_cyclic_weight = config.getfloat(model_type, 's2t_cyclic_weight')
    t2s_cyclic_weight = config.getfloat(model_type, 't2s_cyclic_weight')
    s2t_task_weight = config.getfloat(model_type, 'task_weight')
    t2s_task_weight = config.getfloat(model_type, 't2s_task_weight')
    s2t_style_weight = config.getfloat(model_type, 's2t_style_weight')
    t2s_style_weight = config.getfloat(model_type, 't2s_style_weight')
    discriminator_step = config.getint(model_type, 'discriminator_step')
    generator_step = config.getint(model_type, 'generator_step')
    save_dir = os.path.join(log_dir, utils.make_savedir(config))
#    save_dir = os.path.join(log_dir, config.get('config', 'savedir'))

    if args.delete and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
	
    os.makedirs(save_dir, exist_ok=True)

    model_path = importlib.import_module(model_type)
    model = getattr(model_path, 'model')

    da_model = model(args, config)

    writer = tf.summary.FileWriter(save_dir, sess.graph)
    global_step = tf.train.get_or_create_global_step()

    get_batches = getattr(dataset_utils, model_type)

    tf.logging.info('Training %s with %s' % (model_type, adversarial_mode))
    
    if model_type == 'da_cil':
        with tf.name_scope(model_type + '_batches'):
            source_image_batch, source_label_batch, source_measure_batch, source_command_batch = get_batches('source', 'train', tfrecord_dir, batch_size=args.batch_size, config=config, args=args)
    
        if source_only:
            da_model(source_image_batch, None, source_measure_batch)
        else:       
            target_image_batch, _, _, _ = get_batches('target', 'train', tfrecord_dir, batch_size=args.batch_size, config=config, args=args)
            da_model(source_image_batch, target_image_batch, source_measure_batch)

        with tf.name_scope(model_type + '_objectives'):
            da_model.create_objective(source_label_batch, source_command_batch, adversarial_mode)

            if source_only:
                discriminator_loss = da_model.task_loss
                da_model.summary['discriminator_loss'] = discriminator_loss
            else:
                generator_loss = s2t_cyclic_weight * da_model.s2t_cyclic_loss + t2s_cyclic_weight * da_model.t2s_cyclic_loss + da_model.s2t_adversarial_loss[0] + da_model.t2s_adversarial_loss[0]
                generator_loss += s2t_style_weight * da_model.s2t_style_loss + t2s_style_weight * da_model.t2s_style_loss
                da_model.summary['generator_loss'] = generator_loss

                discriminator_loss = s2t_adversarial_weight * da_model.s2t_adversarial_loss[1] + t2s_adversarial_weight * da_model.t2s_adversarial_loss[1] + s2t_task_weight * da_model.task_loss 
                if t2s_task:
                    discriminator_loss += t2s_task_weight * da_model.t2s_task_loss 
                da_model.summary['discriminator_loss'] = discriminator_loss

    elif model_type == 'pixel_da':
        with tf.name_scope(model_type + '_batches'):
            source_image_batch, source_label_batch = get_batches('source', 'train', tfrecord_dir, batch_size=args.batch_size, config=config)
            mask_image_batch = source_image_batch[:,:,:,3]
            source_image_batch = source_image_batch[:,:,:,:3]
            if config.getboolean(model_type, 'input_mask'):
                tf.logging.info('Using masked input')
                mask_images = tf.to_float(tf.greater(mask_image_batch, 0.9))
                source_image_batch = tf.multiply(source_image_batch, tf.tile(tf.expand_dims(mask_images, 3), [1,1,1,3])) 
                
            # Label is already an 1-hot labels, but we expect categorical
            source_label_max_batch = tf.argmax(source_label_batch, 1)
            source_lateral_label_batch = (source_label_max_batch % 9) / 3
            source_head_label_batch = source_label_max_batch % 3
            
            target_image_batch, _ = get_batches('target', 'train', tfrecord_dir, batch_size=args.batch_size, config=config)

        da_model(source_image_batch, target_image_batch)

        with tf.name_scope(model_type + '_objectives'):
            da_model.create_objective(source_head_label_batch, source_lateral_label_batch, adversarial_mode)

            generator_loss = s2t_cyclic_weight * da_model.s2t_cyclic_loss + t2s_cyclic_weight * da_model.t2s_cyclic_loss + da_model.s2t_adversarial_loss[0] + da_model.t2s_adversarial_loss[0]
            generator_loss += s2t_style_weight * da_model.s2t_style_loss + t2s_style_weight * da_model.t2s_style_loss
            da_model.summary['generator_loss'] = generator_loss

            discriminator_loss = s2t_adversarial_weight * da_model.s2t_adversarial_loss[1] + t2s_adversarial_weight * da_model.t2s_adversarial_loss[1] + s2t_task_weight * da_model.transferred_task_loss 
            if t2s_task:
                discriminator_loss += t2s_task_weight * da_model.t2s_task_loss 
            da_model.summary['discriminator_loss'] = discriminator_loss

    else:
        raise Exception('Not supported model')

    with tf.name_scope('optimizer'):
        tf.logging.info('Getting optimizer')
        if args.lr_decay:
            decay_steps = config.getint('optimizer', 'decay_steps')
            decay_rate = config.getfloat('optimizer', 'decay_rate')
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        else:
            learning_rate = args.learning_rate

        if not source_only:
            g_optimizer = _get_optimizer(config, args.optimizer)(learning_rate)
            g_optim = _gradient_clip(name='generator', optimizer=g_optimizer, loss=generator_loss, global_steps=global_step, clip_norm=args.clip_norm)
        d_optimizer = _get_optimizer(config, args.optimizer)(learning_rate)
        d_optim = _gradient_clip(name='discriminator', optimizer=d_optimizer, loss=discriminator_loss, global_steps=global_step, clip_norm=args.clip_norm)
    if not source_only:
        generator_summary, discriminator_summary = utils.summarize(da_model.summary, t2s_task) 
        utils.config_summary(save_dir, s2t_adversarial_weight, t2s_adversarial_weight, s2t_cyclic_weight, t2s_cyclic_weight, s2t_task_weight, t2s_task_weight, discriminator_step, generator_step, adversarial_mode, whether_noise, noise_dim, s2t_style_weight, t2s_style_weight)
    else:
        discriminator_summary = utils.summarize(da_model.summary, t2s_task, source_only)


    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    if args.load_ckpt:
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for iter_count in range(args.max_iter):
            # Update discriminator
            for disc_iter in range(discriminator_step):
                d_loss, _, steps = sess.run([discriminator_loss, d_optim, global_step])
                if not source_only and adversarial_mode == 'FISHER':
                    _, _ = sess.run([da_model.s2t_adversarial_loss[-1], da_model.t2s_adversarial_loss[-1]])
                #writer.add_summary(disc_sum, steps)
                tf.logging.info('Step %d: Discriminator loss=%.5f', steps, d_loss)

            if not source_only:
                for gen_iter in range(generator_step):
                    g_loss, _, steps = sess.run([generator_loss, g_optim, global_step])
                    #writer.add_summary(gen_sum, steps)
                    tf.logging.info('Step %d: Generator loss=%.5f', steps, g_loss)
            
            if (iter_count+1) % args.save_interval == 0:
                saver.save(sess, os.path.join(save_dir, model_type), global_step=(iter_count+1))
                tf.logging.info('Checkpoint save')

            if (iter_count+1) % args.summary_interval == 0:
                if not source_only:
                    disc_sum, gen_sum = sess.run([discriminator_summary, generator_summary])
                    writer.add_summary(gen_sum, steps)
                else:
                    disc_sum = sess.run(discriminator_summary)
                    
                writer.add_summary(disc_sum, steps)
                tf.logging.info('Summary at %d step' % (iter_count+1))

        
    except tf.errors.OutOfRangeError:
        print('Epoch limited')
    except KeyboardInterrupt:
        print('End training')
    finally:
        coord.request_stop()
        coord.join(threads)      


