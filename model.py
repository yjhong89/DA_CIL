import tensorflow as tf
import numpy as np
import modules
import utils
import dataset_utils

class model():
    def __init__(self, sess, args, config):
        self.sess = sess
        self.args = args
        self.config = config

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
        self.regression = modules.task_regression(regression_channel, image_fc, measurement_fc, command_fc, dropout) 
        
        self.end_point = dict()


    # Build model
    def __call__(self, source, target, measurements):
        with tf.name_scope('generator'):
            self.end_point['source_image'] = source
            self.end_point['target_image'] = target
            if generator_type == 'DRN':
                self.g_s2t = self.generator.generator_drn(source, name='G_S2T')
                # [batch, height, width, 3]
                self.end_point['source_transferred'] = self.g_s2t                
                self.g_t2s = self.generator.generator_drn(target, name='G_T2S')
                self.end_point['target_transferred'] = self.g_t2s

                self.s2t2s = self.generator.generator_drn(self.g_s2t, reuse=True, name='G_T2S')
                self.end_point['back2source'] = self.s2t2s
                self.t2s2t = self.generator.generator_drn(self.g_t2s, reuse=True, name='G_S2T')
                self.end_point['back2target'] = self.t2s2t
            elif generator_type == 'UNET':
                self.g_s2t = self.generator.generator_unet(fake_image, name='G_S2T')
                self.end_point['source_transferred'] = self.g_s2t
                self.g_t2s = self.generator.generator_unet(real_image, name='G_T2S')    
                self.end_point['target_transferred'] = self.g_t2s

                self.s2t2s = self.generator.generator_unet(source, reuse=True, name='G_T2S')
                self.end_point['back2source'] = self.s2t2s
                self.t2s2t = self.generator.generator_unet(target, reuse=True, name='G_S2T') 
                self.end_point['back2target'] = selef.t2s2t
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

    def create_objective(self, steer, acceleration):
        with tf.name_scope('cyclic'):
            self.s2t_cyclic_loss = tf.reduce_mean(tf.abs(self.end_point['source_image'] - self.s2t2s))
            self.t2s_cyclic_loss = tf.reduce_mean(tf.abs(self.end_point['target_image'] - self.t2s2t))
            self.end_point['cyclic_loss'] = self.s2t_cyclic_loss + self.t2s_cyclic_loss
        
        # Wasserstein with gradient-penalty
        with tf.name_scope('adversarial'):
            def gp(real, fake, name):
                epsilon = tf.random_uniform([], maxval=1.0)
                x_hat = epsilon * real + (1 - epsilon) * fake
                # 70x70 patch GAN
                discriminator_logits = tf.reduce_mean(self.discriminator(x_hat, reuse=True, name=name))
                # tf.gradient returns list of sum dy/dx for each xs in x
                gradient = tf.gradient(discriminator_logit, trainable_vars)[0]
                gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.sqruare(gradient - 1), axis=[1,2,3]))
                # Expectation over batches
                gradient_penalty = tf.reduce_mean(tf.sqruare(gradient_l2_norm - 1.0))

                return gradient_penalty

            self.s2t_g_loss = -tf.reduce_mean(self.s2t_fake)
            s2t_d_real_loss = tf.reduce_mean(self.target_real)
            s2t_d_fake_loss = tf.reduce_mean(self.s2t_fake)
            s2t_gp = gp(self.target_real, self.s2t_fake, name='D_S2T')
            self.s2t_d_loss = s2t_d_fake_loss - s2t_d_real_loss + self.args.gp_lambda * s2t_gp
            self.end_point['s2t_d_loss'] = self.s2t_d_loss
           
            self.t2s_g_loss = -tf.reduce_mean(self.t2s_fake)
            t2s_d_real_loss = tf.reduce_mean(self.source_real)
            t2s_d_fake_loss = tf.reduce_mean(self.t2s_fake)
            t2s_gp = gp(self.source_real, self.t2s_fake, name='D_T2S')
            self.t2s_d_loss = t2s_d_fake_loss - t2s_d_real_loss + self.args.gp_lambda * t2s_gp
            self.end_point['t2s_d_loss'] = self.t2s_d_loss

        with tf.name_scope('task'):
            # stats = steer, speed, acc
            

        
