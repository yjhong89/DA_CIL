import tensorflow as tf
import tensorflow.contrib.slim as slim


def cyclic_loss(origin, back2origin):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(origin - back2origin), [1,2,3]))

def adversarial_loss(real_sample, fake_sample, real_logits, fake_logits, discriminator, mode='WGP', discriminator_name='D_S2T', gp_lambda=10):
    def gp(real, fake, name):
        epsilon = tf.random_uniform([], maxval=1.0)
        x_hat = epsilon * real + (1 - epsilon) * fake
        # 70x70 patch GAN
        discriminator_logits = discriminator(x_hat, reuse=True, name=name)
        # tf.gradient returns list of sum dy/dx for each xs in x
        gradient = tf.gradients(discriminator_logits, [x_hat])[0]
        gradient_l2_norm = tf.sqrt(tf.square(gradient))
        # Expectation over batches
        gradient_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradient_l2_norm - 1.0), axis=[1,2,3]))

        return gradient_penalty

    if mode == 'WGP':
        g_loss = -tf.reduce_mean(tf.reduce_sum(fake_logits, [1,2,3]))

        d_real_loss = tf.reduce_mean(tf.reduce_sum(real_logits, [1,2,3]))
        d_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_logits, [1,2,3]))
        gradient_penalty = gp(real_sample, fake_sample, discriminator_name)
        d_loss = d_fake_loss - d_real_loss + gradient_penalty * gp_lambda

    elif mode == 'LS':
        g_loss = tf.reduce_mean(tf.reduce_sum(tf.square(fake_logits - 1), axis=[1,2,3]))
        d_loss = tf.reduce_mean(tf.reduce_sum(tf.square(real_logits -1), axis=[1,2,3])) \
                    + tf.reduce_mean(tf.reduce_sum(tf.square(fake_logits), axis=[1,2,3]))
    
    return g_loss, d_loss
   
def task_regression_loss(steer, acceleration, command, logits, acc_weight):
    assert isinstance(logits, list)

    # steer, acc_x, acc_y, acc_z
    # Steer: [batch, 1], acc: [batch, 3]
    regression_output_list = list()
    for branch in range(len(logits)):
        steer_output = tf.square(logits[branch][:,:1] - steer)
        acc_output = acc_weight * tf.square(logits[branch][:,1:] - acceleration)
        regression_output = tf.reduce_sum(tf.concat([steer_output, acc_output], axis=-1), axis=-1)
        regression_output_list.append(regression_output)
        
    # [batch, 4]:, 4 represent 4 branch regression output
    regression = tf.stack(regression_output_list, axis=-1)

    # One-hot encoded command
    #self.command = tf.placeholder(tf.int32, [self.args.batch_size, 4], name='command')
    # Mask with command
    regression_loss = tf.reduce_mean(command * regression)

    return regression_loss

def task_classifier_loss(head_labels, lateral_labels, head_logits, lateral_logits, weight, num_classes=3):
    total_loss = 0
    lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(lateral_labels, tf.int64), num_classes)
    head_one_hot_labels = slim.one_hot_encoding(tf.cast(head_labels, tf.int64), num_classes)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=lateral_one_hot_labels, logits=lateral_logits, weights=weight)
    total_loss += loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=head_one_hot_labels, logits=head_logits, weights=weight)
    total_loss += loss

    return total_loss
