import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def cyclic_loss(origin, back2origin):
    # Need to scale between loss terms
    return tf.reduce_mean(tf.abs(origin-back2origin))
    #return tf.reduce_mean(tf.reduce_sum(tf.abs(origin - back2origin), [1,2,3]))

def adversarial_loss(real_sample, fake_sample, real_logits, fake_logits, discriminator, mode='WGP', discriminator_name='D_S2T', gp_lambda=10, rho=1e-5, patch=True):
    def gp(real, fake, name):
        batch_size, _,_,_ = real.get_shape().as_list()
        # Get different epsilon for each samples in a batch
        epsilon = tf.random_uniform([batch_size, 1, 1, 1], maxval=1.0)
        x_hat = epsilon * real + (1 - epsilon) * fake
        # 70x70 patch GAN, [batch size, 12, 20, 1]
        tf.logging.info('Gradient penalty of %s' % name)
        discriminator_logits, _ = discriminator(x_hat, patch=patch, reuse=True, name=name)
        d_logits = tf.reshape(discriminator_logits, [batch_size, -1])
        _, num_cols = d_logits.get_shape().as_list()
        '''
            TypeError: 'Tensor' object is not iterable
            Use tf.while_loop(condition, body, loop_vars)
        '''
        each_logits = tf.unstack(d_logits, axis=1)        
        # Track both the loop index and summation in a tuple form
            # Summation would be float32
        index_summation = (tf.constant(0), tf.constant(0.0))
        # The loop condition, i < 12*20
        def condition(index, summation):
            return tf.less(index, batch_size)

        def body(index, summation):
            gradient_i = tf.gather(each_logits, index)
            # gradient summation for different samples in a batch, differentiation would be 0 for diffenent samples because they are independent
                # d(D(x2))/dx1 = 0
            gradient = tf.gradients(gradient_i, [x_hat])[0]
            gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1,2,3]))
            gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1.0))
            # Reduce mean
            return tf.add(index, 1), tf.add(summation, gradient_penalty/num_cols)

        return tf.while_loop(condition, body, index_summation)[1]

#        # tf.gradient returns list of sum dy/dx for each xs in x
#        gradient = tf.gradients(discriminator_logits, [x_hat])[0]
#        gradient_l2_norm = tf.sqrt(tf.square(gradient))
#        # Expectation over batches
#        gradient_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradient_l2_norm - 1.0), axis=[1,2,3]))
#
#        return gradient_penalty

    if mode == 'WGP':
#        g_loss = -tf.reduce_mean(tf.reduce_sum(fake_logits, [1,2,3]))
#
#        d_real_loss = tf.reduce_mean(tf.reduce_sum(real_logits, [1,2,3]))
#        d_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_logits, [1,2,3]))
#        gradient_penalty = gp(real_sample, fake_sample, discriminator_name)
#        d_loss = d_fake_loss - d_real_loss + gradient_penalty * gp_lambda
        g_loss = -tf.reduce_mean(fake_logits)

        d_real_loss = tf.reduce_mean(real_logits)
        d_fake_loss = tf.reduce_mean(fake_logits)
        gradient_penalty = gp(real_sample, fake_sample, discriminator_name)
        d_loss = d_fake_loss - d_real_loss + gradient_penalty * gp_lambda

    elif mode == 'LS':
#        g_loss = tf.reduce_mean(tf.reduce_sum(tf.square(fake_logits - 1), axis=[1,2,3]))
#        d_loss = tf.reduce_mean(tf.reduce_sum(tf.square(real_logits - 1), axis=[1,2,3])) \
#                    + tf.reduce_mean(tf.reduce_sum(tf.square(fake_logits), axis=[1,2,3]))
        # g_loss = tf.reduce_mean((tf.square(tf.nn.sigmoid(fake_logits) - 1)))
        g_loss = -tf.reduce_mean(tf.square(tf.sigmoid(fake_logits)))
        d_loss = tf.reduce_mean(tf.square(tf.nn.sigmoid(real_logits) - 1)) \
                    + tf.reduce_mean(tf.square(tf.nn.sigmoid(fake_logits)))
    elif mode == 'FISHER':
        with tf.variable_scope(discriminator_name):
            alpha = tf.get_variable('fisher_lambda', [], initializer=tf.constant_initializer(0))
        
        e_q_f = tf.reduce_mean(fake_logits)
        e_p_f = tf.reduce_mean(real_logits)
        e_q_f2 = tf.reduce_mean(tf.square(fake_logits))
        e_p_f2 = tf.reduce_mean(tf.square(real_logits))

        constraint = 1 - 0.5 * (e_p_f2 + e_q_f2)

        g_loss = -tf.reduce_mean(fake_logits)
        d_loss = -1 * (e_p_f - e_q_f + alpha*constraint - rho/2 * constraint**2)

        alpha_opt = tf.train.GradientDescentOptimizer(rho).minimize(-d_loss, var_list=[alpha])

        return g_loss, d_loss, alpha_opt

    else:
        raise ValueError('%s is not supported' % mode)
    
    return g_loss, d_loss

def style_loss(fake_activations, real_activations, weights):
    
    assert len(fake_activations) == len(real_activations) == len(weights)   

    total_loss = 0
 
    # [channel * channel] matrix
    def gram_mtx(tensor):
        shape = tensor.get_shape().as_list()
        
        channel_vector = tf.reshape(tensor, [-1, shape[3]])
        gram = tf.matmul(tf.transpose(channel_vector), channel_vector)

        N = np.prod(shape[1:])
        
        return gram, N, shape[0]

    for i in range(len(fake_activations)):
        fake_gram, fake_N, fake_batch_num = gram_mtx(fake_activations[i])
        real_gram, real_N, real_batch_num = gram_mtx(real_activations[i])
        
        assert fake_N == real_N
        assert fake_batch_num == real_batch_num

        loss = (1.0 / ((fake_batch_num**2) * (fake_N**2))) * tf.reduce_sum(tf.square(fake_gram - real_gram))

        total_loss += float(weights[i]) * loss

    return total_loss

   
def task_regression_loss(steer, command, logits):
    assert isinstance(logits, list)

    # steer, acc_x, acc_y, acc_z
    # Steer: [batch, 1], acc: [batch, 3]
    regression_output_list = list()
    for branch in range(len(logits)):
        diff = tf.square(logits[branch][:,:1] - tf.expand_dims(steer, 1))
        print(diff.get_shape().as_list())
        #regression_output = tf.reduce_sum(diff, axis=1)
        regression_output_list.append(diff)
        
    # [batch, 4]:, 4 represent 4 branch regression output
    regression = tf.concat(regression_output_list, axis=1)
    print(regression.get_shape().as_list())

    # Mask with command
    regression_loss = tf.reduce_mean(command * regression)

    return regression_loss

# steer_label is class (one_hot)
def task_classifier_loss(steer_label, command, logits, num_classes=5):
    assert isinstance(logits, list)
    assert len(steer_label.get_shape().as_list()) == 2

    classifier_output_list = list()
    for branch in range(len(logits)):
        # [batch, ]
        # Label smoothing
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=steer_label, logits=logits[branch])        
        classifier_output_list.append(tf.expand_dims(loss, 1))

    classification = tf.concat(classifier_output_list, axis=1)

    # Mask with comamnd
    classification_loss = tf.reduce_mean(command * classification)
    #classification_loss = tf.reduce_mean(classifier_output_list[1])
    
    loss = classification_loss
    #loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    return loss, classification


def task_classifier_pixel_da_loss(head_labels, lateral_labels, head_logits, lateral_logits, num_classes=3):
    total_loss = 0
    lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(lateral_labels, tf.int64), num_classes)
    head_one_hot_labels = slim.one_hot_encoding(tf.cast(head_labels, tf.int64), num_classes)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lateral_one_hot_labels, logits=lateral_logits))
    total_loss += loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=head_one_hot_labels, logits=head_logits))
    total_loss += loss

    return total_loss
