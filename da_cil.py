import tensorflow as tf
import numpy as np
import modules
import utils
import dataset_utils

class model():
    def __init__(self, args, config):
        self.args = args

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

        self.acc_hparam = config.getfloat('model', 'acc_hparam')
        self.adversarial_hparam = config.getfloat('model', 'adversarial_hparam')
        self.regression_hparam = config.getfloat('model', 'regression_hparam')
        self.cyclic_hparam = config.getfloat('model', 'cyclic_hparam')

        self.t2s_regression = config.getboolean('model', 't2s_regression')
        self.t2s_regression_hparam = config.getboolean('model', 't2s_regression')
        
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
            if self.t2s_regression:
                self.t2s_end = self.regression(self.g_t2s, measurements, reuse=True)

    def create_objective(self, steer, acceleration, command):
        self.command = command
        with tf.name_scope('cyclic'):
            self.s2t_cyclic_loss = self._cyclic_loss(self.summary['source_image'], self.s2t2s)
            self.t2s_cyclic_loss = self._cyclic_loss(self.summary['target_image'], self.t2s2t)
            self.summary['cyclic_loss'] = self.s2t_cyclic_loss + self.t2s_cyclic_loss
        
        # Wasserstein with gradient-penalty
        with tf.name_scope('adversarial'):
            self.s2t_g_loss, self.s2t_d_loss = self._adversarial_loss(self.target_real, self.s2t_fake, name='D_S2T', mode='WGP')
            self.t2s_g_loss, self.t2s_d_loss = self._adversarial_loss(self.source_real, self.t2s_fake, name='D_T2S', mode='WGP')
            self.summary['s2t_d_loss'] = self.s2t_d_loss
            self.summary['t2s_d_loss'] = self.t2s_d_loss            

        with tf.name_scope('task'):
            self.regression_loss = self._task_regression_loss(steer, acceleration, command, self.end)
            self.summary['regression_loss'] = self.regression_loss
            if self.t2s_regression:
                self.t2s_regression_loss = seef._task_regression_loss(steer, acceleration, command, self.t2s_end)
                self.summary['t2s_regression_loss'] = self.t2s_regression_loss


    def _cyclic_loss(self, origin, back2origin):
        return tf.reduce_mean(tf.abs(origin - back2origin))

    def _adversarial_loss(self, real, fake, name, mode='WGP')
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

        g_loss = -tf.reduce_mean(fake)

        d_real_loss = tf.reduce_mean(real)
        d_fake_loss = tf.reduce_mean(fake)
        gradient_penalty = gp(real, fake, name)
        d_loss = d_real_loss + d_fake_loss * selef.args.gp_lambda

        return g_loss, d_loss
       
    def _task_regression_loss(self, steer, acceleration, command, logits):
        assert isinstance(logits, list)

        # steer, acc_x, acc_y, acc_z
        # Steer: [batch, 1], acc: [batch, 3]
        regression_output_list = list()
        for branch in range(len(logits)):
            steer_output = tf.square(logits[branch][:,:1] - steer)
            acc_output = self.acc_hparam * tf.square(logits[branch][:,1:] - acceleration)
            regression_output = tf.reduce_sum(tf.concat([steer_output, acc_output], axis=-1), axis=-1)
            regression_output_list.append(regression_output)
            
        # [batch, 4]:, 4 represent 4 branch regression output
        regression = tf.stack(regression_output_list, axis=-1)

        # One-hot encoded command
        #self.command = tf.placeholder(tf.int32, [self.args.batch_size, 4], name='command')
        # Mask with command
        regression_loss = tf.reduce_mean(command * regression)

        return regression_loss
        
    def g_step_loss(self):
        generator_loss = 0
        # source 2 target
        generator_loss += self.s2t_g_loss * self.adversarial_hparam 
        # target 2 source
        generator_loss += self.t2s_g_loss * self.adversarial_hparam
        # cyclic consistency
        generator_loss += self.s2t_cyclic_loss * self.cyclic_hparam
        generator_loss += self.t2s_cyclic_loss * self.cyclic_hparam
        self.summary['generator_loss'] = generator_loss

        return generator_loss

    def d_step_loss(self):
        discriminator_loss = 0
        # target discriminator
        discriminator_loss += self.s2t_d_loss * self.adversarial_hparam
        # source discriminator 
        discriminator_loss += self.t2s_d_loss * self.adversarial_hparam
        # task regression, not directly connected to disicriminator
        discriminator_loss += self.regression_loss * self.regression_hparam

        if self.t2s_regression:
            discriminator_loss += self.t2s_regression_loss * self.t2s_regression_hparam

        self.summary['discriminator_loss'] = discriminator_loss

        return discriminator_loss
        
