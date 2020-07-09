import tensorflow as tf

# TODO: can 'is_train' be removed? (along with Dropout layer)

def resnet(inputs, num_blocks, is_train):

    n2d_filters  = 64
    window2d     = 3

    activation = tf.nn.elu
    conv1d = tf.layers.conv1d
    conv2d = tf.layers.conv2d

    # project features down to n2d_filters
    layers2d = [inputs]
    layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
    layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
    layers2d.append(activation(layers2d[-1]))

    # 2D ResNet with dilations
    dilation = 1
    for _ in range(num_blocks):
        layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
        layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1]))
        layers2d.append(tf.keras.layers.Dropout(rate=0.15)(layers2d[-1], training=is_train))
        layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
        layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1] + layers2d[-7]))
        dilation *= 2
        if dilation > 16:
            dilation = 1

    # logits for theta
    logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')

    # logits for phi
    logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')

    # symmetrize
    layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

    # logits for distances
    logits_dist = conv2d(layers2d[-1], 37, 1, padding='SAME')

    # logits for beta-strand pairings
    logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')

    # logits for omega
    logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')

    return logits_dist,logits_omega,logits_theta,logits_phi
