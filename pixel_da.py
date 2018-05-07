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
        discriminator_channel = config.getint('discriminator', 'channel')
        generator_channel = config.getint('generator', 'channel')
        classifier_channel = config.getint('task', 'channel')

        self.generator_type = config.get('generator', 'type')
        self.generator_out_channel = config.getint('generator', 'out_channel')
        self.generator = modules.generator(generator_channel, config, self.args)
        self.discriminator = modules.discriminator(discriminator_channel, group_size=self.args.batch_size)
        self.patch = config.getboolean('discriminator', 'patch')
        self.discriminator_dropout_prob = config.getfloat('discriminator', 'dropout_prob')    
        self.t2s_task = config.getboolean('config', 't2s_task')            

        self.style_weights = config.getlist('discriminator', 'style_weights')

        self.transferred_classifier = modules.task_classifier(classifier_channel, num_classes=3, training=self.args.training) 

        self.transferred_task_weight = config.getfloat(model_name, 'task_weight')
        self.t2s_task_weight = config.getfloat(model_name, 't2s_task_weight')

        self.summary = dict()


    # Build model
    def __call__(self, source, target):
        with tf.name_scope('generator'):
            self.summary['source_image'] = source
            self.summary['target_image'] = target
            tf.logging.info('Use %s architecture' % self.generator_type)
            try:
                generator_func = getattr(self.generator, self.generator.module_name + '_' + self.generator_type.lower())
            except:
                raise AttributeError('%s not supproted' % self.generator_type)

            self.g_s2t, self.source_noise = generator_func(source, name='G_S2T')
            self.g_t2s, self.target_noise = generator_func(target, name='G_T2S')
            self.s2t2s, _ = generator_func(self.g_s2t[:,:,:,:3], reuse=True, name='G_T2S')
            self.t2s2t, _ = generator_func(self.g_t2s[:,:,:,:3], reuse=True, name='G_S2T')

            self.summary['source_transferred'] = self.g_s2t[:,:,:,:3]                
            self.summary['target_transferred'] = self.g_t2s[:,:,:,:3]
            self.summary['back2source'] = self.s2t2s[:,:,:,:3]
            self.summary['back2target'] = self.t2s2t[:,:,:,:3]
            

        with tf.name_scope('discriminator'):
            # Patch discriminator
            self.s2t_fake, self.s2t_fake_activations = self.discriminator(self.g_s2t[:,:,:,:3], name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
            self.t2s_fake, self.t2s_fake_activations = self.discriminator(self.g_t2s[:,:,:,:3], name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
    
            self.target_real, self.target_real_activations = self.discriminator(target, reuse=True, name='D_S2T', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)
            self.source_real, self.source_real_activations = self.discriminator(source, reuse=True, name='D_T2S', dropout_prob=self.discriminator_dropout_prob, patch=self.patch)

        with tf.name_scope('classifier'):
            self.head_logits, self.lateral_logits = self.transferred_classifier(self.g_s2t, reuse_private=False, reuse_shared=False, shared='transferred_shared', private='transferred_private')  
            if self.t2s_task:
                self.t2s_head_logits, self.t2s_lateral_logits = self.transferred_classifier(self.g_t2s, reuse_private=False, reuse_shared=True, shared='transferred_shared', private='t2s_private')

            # Evaluation
            if not self.args.training:
                self.target_head_logits, self.target_lateral_logits = self.transferred_classifier(self.summary['target_image'], reuse_private=True, reuse_shared=True, shared='transferred_shared', private='transferred_private')
                self.soure_head_logits, self.source_lateral_logits = self.transferred_classifier(self.summary['source_image'], reuse_private=True, reuse_shared=True, shared='transferred_shared', private='t2s_private')

    def create_objective(self, head_labels, lateral_labels, mode='LS'):
        with tf.name_scope('cyclic'):
            self.s2t_cyclic_loss = losses.cyclic_loss(self.summary['source_image'], self.s2t2s)
            self.t2s_cyclic_loss = losses.cyclic_loss(self.summary['target_image'], self.t2s2t)
                
            self.summary['source_cyclic_loss'] = self.s2t_cyclic_loss
            self.summary['target_cyclic_loss'] = self.t2s_cyclic_loss
        
        # Wasserstein with gradient-penalty
        with tf.name_scope('adversarial'):
            self.s2t_adversarial_loss = losses.adversarial_loss(self.summary['target_image'], self.g_s2t, self.target_real, self.s2t_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_S2T')
            self.t2s_adversarial_loss = losses.adversarial_loss(self.summary['source_image'], self.g_t2s, self.source_real, self.t2s_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_T2S')

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
            self.transferred_task_loss = losses.task_classifier_pixel_da_loss(head_labels, lateral_labels, self.head_logits, self.lateral_logits)
            self.summary['task_loss'] = self.transferred_task_loss
            if self.t2s_task:
                self.t2s_task_loss = losses.task_classifier_pixel_da_loss(head_labels, lateral_labels, self.t2s_head_logits, self.t2s_lateral_logits)
                self.summary['t2s_task_loss'] = self.t2s_task_loss
       

