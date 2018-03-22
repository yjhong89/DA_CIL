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
        generator_type = config.get('generator', 'type')
        generator_channel = config.getint('generator', 'channel')
        classifier_channel = config.getint('classifier', 'channel')
        dropout = config.getfloat('classifier', 'dropout')

        self.generator = modules.generator(generator_channel)
        self.discriminator = modules.discriminator(discriminator(channel)
        
        self.transferred_classifier = modules.task_classifier(classifier_channel, num_classes=3, training=self.args.training) 

        self.transferred_task_hparam = config.getfloat(model_name, 'transferred_task_hparam')
        self.t2s_task_hparam = config.getfloat(model_name, 't2s_task_hparam')

        self.summary = dict()


    # Build model
    def __call__(self, source, target):
        with tf.name_scope('generator'):
            self.summary['source_image'] = source
            self.summary['target_image'] = target
            if generator_type == 'DRN':
                self.g_s2t = self.generator.generator_drn(source, name='G_S2T')
                # [batch, height, width, 3]
                self.summary['source_transferred'] = self.g_s2t                
                self.g_t2s = self.generator.generator_drn(target, name='G_T2S')
                self.summary['target_transferred'] = self.g_t2s

                self.s2t2s = self.generator.generator_drn(self.g_s2t, reuse=True, name='G_T2S')
                self.summary['back2source'] = self.s2t2s
                self.t2s2t = self.generator.generator_drn(self.g_t2s, reuse=True, name='G_S2T')
                self.summary['back2target'] = self.t2s2t
            elif generator_type == 'UNET':
                self.g_s2t = self.generator.generator_unet(fake_image, name='G_S2T')
                self.summary['source_transferred'] = self.g_s2t
                self.g_t2s = self.generator.generator_unet(real_image, name='G_T2S')    
                self.summary['target_transferred'] = self.g_t2s

                self.s2t2s = self.generator.generator_unet(source, reuse=True, name='G_T2S')
                self.summary['back2source'] = self.s2t2s
                self.t2s2t = self.generator.generator_unet(target, reuse=True, name='G_S2T') 
                self.summary['back2target'] = selef.t2s2t
            else:
                raise Exception('Not supported type')

        with tf.name_scope('discriminator'):
            # Patch discriminator
            self.s2t_fake = self.discriminator(self.g_s2t, name='D_S2T')
            self.t2s_fake = self.discriminator(self.g_t2s, name='D_T2S')

            self.target_real = self.discriminator(target, reuse=True, name='D_S2T')
            self.source_real = self.discriminator(source, reuse=True, name='D_T2S')

        with tf.name_scope('classifier'):
            self.head_logits, self.lateral_logits = self.transferred_classifier(self.g_s2t, reuse_private=False, reuse_shared=False, shared='transferred_shared', private='transferred_shared')  
            if self.t2s_task:
                self.t2s_head_logits, self.t2s_lateral_logits = self.transferred_classifier(self.g_t2s, reuse_private=False, reuse_shared=True, shared='transferred_shared', private='t2s_private')

    def create_objective(self, head_labels, lateral_labels):
        self.command = command
        with tf.name_scope('cyclic'):
            self.s2t_cyclic_loss = losses.cyclic_loss(self.summary['source_image'], self.s2t2s)
            self.t2s_cyclic_loss = losses.cyclic_loss(self.summary['target_image'], self.t2s2t)
            self.summary['cyclic_loss'] = self.s2t_cyclic_loss + self.t2s_cyclic_loss
        
        # Wasserstein with gradient-penalty
        with tf.name_scope('adversarial'):
            self.s2t_g_loss, self.s2t_d_loss = losses.adversarial_loss(self.target_real, self.s2t_fake, name='D_S2T', mode='WGP')
            self.t2s_g_loss, self.t2s_d_loss = losses.adversarial_loss(self.source_real, self.t2s_fake, name='D_T2S', mode='WGP')
            self.summary['s2t_d_loss'] = self.s2t_d_loss
            self.summary['t2s_d_loss'] = self.t2s_d_loss            

        with tf.name_scope('task'):
            self.transferred_task_loss = losses.task_classifier_loss(head_labels, lateral_labels, self.head_logits, self.lateral_logits, weight=self.transferred_task_hparam)
            self.summary['transferred_task_loss'] = self.transferred_task_loss
            if self.args.t2s_task:
                self.t2s_task_loss = losses.task_classifier_loss(head_labels, lateral_labels, self.t2s_head_logits, self.t2s_lateral_logits, weight=self.t2s_task_hparam)
                self.summary['self.t2s_task_loss'] = self.t2s_task_loss
       

