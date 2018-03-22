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
        command_fc = config.getint('regression', 'command_fc')
        dropout = config.getfloat('regression', 'dropout')

        self.generator = modules.generator(generator_channel)
        self.discriminator = modules.discriminator(discriminator(channel)
        # List of 4 branch modules
        self.regression = modules.task_regression(regression_channel, image_fc, measurement_fc, command_fc, dropout) 

        self.acc_weight = config.getfloat(model_name, 'acc_weight')
        self.adversarial_hparam = config.getfloat(model_name, 'adversarial_hparam')
        self.regression_hparam = config.getfloat(model_name, 'regression_hparam')
        self.cyclic_hparam = config.getfloat(model_name, 'cyclic_hparam')
        self.t2s_regression_hparam = config.getfloat(model_name, 't2s_regression')
        
        self.summary = dict()


    # Build model
    def __call__(self, source, target, measurements):
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

        with tf.name_scope('regression'):
            self.end = self.regression(self.g_s2t, measurements)  
            if self.args.t2s_regression:
                self.t2s_end = self.regression(self.g_t2s, measurements, reuse=True)

    def create_objective(self, steer, acceleration, command):
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
            self.regression_loss = losses.task_regression_loss(steer, acceleration, command, self.end, self.acc_weight)
            self.summary['regression_loss'] = self.regression_loss
            if self.t2s_task:
                self.t2s_regression_loss = losses.task_regression_loss(steer, acceleration, command, self.t2s_end, self.acc_weight)
                self.summary['t2s_regression_loss'] = self.t2s_regression_loss


        
