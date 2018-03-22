import tensorflow as tf
import tf.contrib.slim as slim


def cyclic_loss(origin, back2origin):
    return tf.reduce_mean(tf.abs(origin - back2origin))

def adversarial_loss(real, fake, name, mode='WGP')
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

    if mode == 'WGP':
        g_loss = -tf.reduce_mean(fake)

        d_real_loss = tf.reduce_mean(real)
        d_fake_loss = tf.reduce_mean(fake)
        gradient_penalty = gp(real, fake, name)
        d_loss = d_real_loss + d_fake_loss * selef.args.gp_lambda

    return g_loss, d_loss
   
def task_regression_loss(steer, acceleration, command, logits):
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

def task_classifier_loss(head_labels, lateral_labels, head_logits, latarel_logits, weight, num_classes=3):
    total_loss = 0
    lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(lateral_labels, tf.int64), num_classes)
    head_one_hot_labels = slim.one_hot_encoding(tf.cast(head_labels, tf.int64), num_classes
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=lateral_one_hot_labels,
                                        logits=lateral_logits,
                                        weights=weight)
    total_loss += loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=head_one_hot_labels,
                                        logits=head_logits,
                                        weights=weight)
    total_loss += loss

    return total_loss
