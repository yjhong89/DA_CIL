import tensorflow as tf
import numpy as np
import modules
import utils
import dataset_utils
import losses

class model():
    def __init__(self, args, config):
        self.args = args

        model_name = config.get('config', 'experiment')
        task_channel = config.getint('task', 'channel')
        task_module_type = config.get('task', 'classifier_type')
        image_fc = config.getint('task', 'image_fc')
        measurement_fc = config.getint('task', 'measurement_fc')
        branch_fc = config.getint('task', 'branch_fc')

        self.source_only = config.getboolean('config', 'source_only')
        self.t2s_task = config.getboolean('config', 't2s_task')
        self.style_weights = config.getlist('discriminator', 'style_weights')

        self.share_all_image = config.getboolean('generator', 'share_all_image')

        if not self.source_only:
            discriminator_channel = config.getint('discriminator', 'channel')
            self.discriminator_dropout_prob = config.getfloat('discriminator', 'dropout_prob')
            self.patch = config.getboolean('discriminator', 'patch')
            self.generator_type = config.get('generator', 'type')
            generator_channel = config.getint('generator', 'channel')
            self.generator_out_channel = config.getint('generator', 'out_channel')
            self.generator = modules.generator(generator_channel, config, args)
            self.discriminator = modules.discriminator(discriminator_channel, group_size=1)
        # List of 4 branch modules
        self.task = modules.task(task_channel, image_fc, measurement_fc, branch_fc, self.args.training, task_module_type) 

        self.summary = dict()


    # Build model
    def __call__(self, source, target, measurements):
        if not self.source_only:
            with tf.name_scope('generator'):
                self.summary['source_image'] = source
                self.summary['target_image'] = target
                tf.logging.info('Use %s architecture' % self.generator_type)
                try:
                    generator_func = getattr(self.generator, self.generator.module_name + '_' + self.generator_type.lower())
                except:
                    raise AttributeError('%s not supproted' % self.generator_type)
 
                if self.share_all_image:   
                    self.source_concat = tf.concat([source[:,:,:,:3],source[:,:,:,3:6],source[:,:,:,6:]],axis=0)
                    self.target_concat = tf.concat([target[:,:,:,:3],target[:,:,:,3:6],target[:,:,:,6:]],axis=0)
                    self.g_s2t, self.source_noise = generator_func(self.source_concat, name='G_S2T')
                    self.g_t2s, self.target_noise = generator_func(self.target_concat, name='G_T2S')
                    self.s2t2s, _ = generator_func(self.g_s2t[:,:,:,:3], reuse=True, name='G_T2S')
                    self.t2s2t, _ = generator_func(self.g_t2s[:,:,:,:3], reuse=True, name='G_S2T')
                    self.summary['source_transferred'] = tf.concat([self.g_s2t[:self.args.batch_size,:,:,:3],self.g_s2t[self.args.batch_size:2*self.args.batch_size,:,:,:3],self.g_s2t[2*self.args.batch_size    :,:,:,:3]],axis=3)
                    self.summary['target_transferred'] = tf.concat([self.g_t2s[:self.args.batch_size,:,:,:3],self.g_t2s[self.args.batch_size:2*self.args.batch_size,:,:,:3],self.g_t2s[2*self.args.batch_size    :,:,:,:3]],axis=3)
                    self.summary['back2source'] = tf.concat([self.s2t2s[:self.args.batch_size,:,:,:3],self.s2t2s[self.args.batch_size:2*self.args.batch_size,:,:,:3],self.s2t2s[2*self.args.batch_size:,:,:,:    3]],axis=3)
                    self.summary['back2target'] = tf.concat([self.t2s2t[:self.args.batch_size,:,:,:3],self.t2s2t[self.args.batch_size:2*self.args.batch_size,:,:,:3],self.t2s2t[2*self.args.batch_size:,:,:,:    3]],axis=3)
                else:
                    self.g_s2t, self.source_noise = generator_func(source, name='G_S2T')
                    self.g_t2s, self.target_noise = generator_func(target, name='G_T2S')
                    self.s2t2s, _ = generator_func(self.g_s2t[:,:,:,:9], reuse=True, name='G_T2S')
                    self.t2s2t, _ = generator_func(self.g_t2s[:,:,:,:9], reuse=True, name='G_S2T')
        
                    self.summary['source_transferred'] = self.g_s2t[:,:,:,:9]                
                    self.summary['target_transferred'] = self.g_t2s[:,:,:,:9]
                    self.summary['back2source'] = self.s2t2s[:,:,:,:9]
                    self.summary['back2target'] = self.t2s2t[:,:,:,:9]
        
            with tf.name_scope('discriminator'):
                # Patch discriminator
                if self.share_all_image:
                    self.s2t_fake, self.s2t_fake_activations = self.discriminator(self.g_s2t[:,:,:,:3], name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
                    self.t2s_fake, self.t2s_fake_activations = self.discriminator(self.g_t2s[:,:,:,:3], name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
    
                    self.target_real, self.target_real_activations = self.discriminator(self.target_concat, reuse=True, name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
                    self.source_real, self.source_real_activations = self.discriminator(self.source_concat, reuse=True, name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
                
                else:
                    self.s2t_fake = self.discriminator(self.g_s2t[:,:,:,:9], name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
                    self.t2s_fake = self.discriminator(self.g_t2s[:,:,:,:9], name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
    
                    self.target_real = self.discriminator(target, reuse=True, name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
                    self.source_real = self.discriminator(source, reuse=True, name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
        
        else:
            self.summary['source_image'] = source

        with tf.name_scope('task'):
            if self.source_only:
                self.end, self.cnn = self.task(self.summary['source_image'], measurements)
            else:
                self.end, _ = self.task(self.summary['source_transferred'], measurements)  
                if self.t2s_task:
                    self.t2s_end, _ = self.task(source, measurements, private='t2s_private', reuse_shared=True)

    def create_objective(self, steer, command, mode='FISHER'):
        if not self.source_only:
            with tf.name_scope('cyclic'):
                if self.generator_out_channel == 9:
                    self.s2t_cyclic_loss = losses.cyclic_loss(self.summary['source_image'], self.s2t2s)
                    self.t2s_cyclic_loss = losses.cyclic_loss(self.summary['target_image'], self.t2s2t)
                else:
                    self.s2t_cyclic_loss = losses.cyclic_loss(self.source_concat, self.s2t2s)
                    self.t2s_cyclic_loss = losses.cyclic_loss(self.target_concat, self.t2s2t)
                    
                self.summary['source_cyclic_loss'] = self.s2t_cyclic_loss
                self.summary['target_cyclic_loss'] = self.t2s_cyclic_loss
                    
            # Wasserstein with gradient-penalty
            with tf.name_scope('adversarial'):
                self.s2t_adversarial_loss = losses.adversarial_loss(None, None, self.target_real, self.s2t_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_S2T', patch=self.patch)
                self.t2s_adversarial_loss = losses.adversarial_loss(None, None, self.target_real, self.s2t_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_T2S', patch=self.patch)
                self.summary['s2t_g_loss'] = self.s2t_adversarial_loss[0]
                self.summary['t2s_g_loss'] = self.t2s_adversarial_loss[0]
                self.summary['s2t_d_loss'] = self.s2t_adversarial_loss[1]
                self.summary['t2s_d_loss'] = self.t2s_adversarial_loss[1]     

            with tf.name_scope('style'):
                self.s2t_style_loss = losses.style_loss(self.s2t_fake_activations, self.target_real_activations, self.style_weights)
                self.t2s_style_loss = losses.style_loss(self.t2s_fake_activations, self.source_real_activations, self.style_weights)
                self.summary['s2t_style_loss'] = self.s2t_style_loss
                self.summary['t2s_style_loss'] = self.t2s_style_loss

        with tf.name_scope('task'):
            self.task_loss, _ = losses.task_classifier_loss(steer, command, self.end)
            self.summary['task_loss'] = self.classification_loss
            if self.t2s_task:
                self.t2s_task_loss, _ = losses.task_classifier_loss(steer, command, self.t2s_end)
                self.summary['t2s_task_loss'] = self.t2s_classification_loss

        
