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
        regression_channel = config.getint('regression', 'channel')
        image_fc = config.getint('regression', 'image_fc')
        measurement_fc = config.getint('regression', 'measurement_fc')
        branch_fc = config.getint('regression', 'branch_fc')

        self.source_only = config.getboolean('config', 'source_only')

        if not self.source_only:
            self.generator = modules.generator(generator_channel, config, args)
            self.discriminator = modules.discriminator(discriminator_channel)
        # List of 4 branch modules
        self.task = modules.task(regression_channel, image_fc, measurement_fc, branch_fc, self.args.training) 

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
                self.s2t_fake = self.discriminator(self.g_s2t[:,:,:,:3], name='D_S2T')
                self.t2s_fake = self.discriminator(self.g_t2s[:,:,:,:3], name='D_T2S')
    
                self.target_real = self.discriminator(target, reuse=True, name='D_S2T')
                self.source_real = self.discriminator(source, reuse=True, name='D_T2S')
        
        else:
            self.summary['source_image'] = source

        with tf.name_scope('regression'):
            if self.source_only:
                self.end = self.task(self.summary['source_image'], measurements)
            else:
                self.end = self.task(self.g_s2t[:,:,:,:3], measurements)  
                if self.args.t2s_task:
                    self.t2s_end = self.task(self.g_t2s, measurements, private='t2s_private', reuse_shared=True)

                

    def create_objective(self, steer, command):
        if not self.source_only:
            with tf.name_scope('cyclic'):
                if self.generator_out_channel == 4:
                    self.s2t_cyclic_loss = losses.cyclic_loss(tf.concat([self.summary['source_image'], self.source_noise], 3), self.s2t2s)
                    self.t2s_cyclic_loss = losses.cyclic_loss(tf.concat([self.summary['target_image'], self.target_noise], 3), self.t2s2t)
                else:
                    self.s2t_cyclic_loss = losses.cyclic_loss(self.summary['source_image'], self.s2t2s)
                    self.t2s_cyclic_loss = losses.cyclic_loss(self.summary['target_image'], self.t2s2t)
                self.summary['cyclic_loss'] = self.s2t_cyclic_loss + self.t2s_cyclic_loss
            
            # Wasserstein with gradient-penalty
            with tf.name_scope('adversarial'):
                if mode == 'FISHER':
                    self.s2t_g_loss, self.s2t_d_loss, self.s2t_alpha = losses.adversarial_loss(self.summary['target_image'], self.g_s2t, self.target_real, self.s2t_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_S2T')
                    self.t2s_g_loss, self.t2s_d_loss, self.t2s_alpha = losses.adversarial_loss(self.summary['source_image'], self.g_t2s, self.source_real, self.t2s_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_T2S')
                else:
                    self.s2t_g_loss, self.s2t_d_loss = losses.adversarial_loss(self.summary['target_image'], self.g_s2t, self.target_real, self.s2t_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_S2T')
                    self.t2s_g_loss, self.t2s_d_loss = losses.adversarial_loss(self.summary['source_image'], self.g_t2s, self.source_real, self.t2s_fake, mode=mode, discriminator=self.discriminator, discriminator_name='D_T2S')
                self.summary['s2t_g_loss'] = self.s2t_g_loss
                self.summary['t2s_g_loss'] = self.t2s_g_loss
                self.summary['s2t_d_loss'] = self.s2t_d_loss
                self.summary['t2s_d_loss'] = self.t2s_d_loss            

        with tf.name_scope('task'):
            self.classification_loss = losses.task_classifier_loss(steer, command, self.end)
            self.summary['classification_loss'] = self.classification_loss
            if self.args.t2s_task:
                self.t2s_classifier_loss = losses.task_classifier_loss(steer, acceleration, command, self.t2s_end, self.acc_weight)
                self.summary['t2s_regression_loss'] = self.t2s_classifier_loss


        
