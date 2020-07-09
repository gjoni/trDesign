import tensorflow as tf
from utils import *

def get_background(msa):

    # network params
    n2d_layers   = 36
    n2d_filters  = 64
    window2d     = 3

    activation = tf.nn.elu
    conv1d = tf.layers.conv1d
    conv2d = tf.layers.conv2d

    BDIR = "/home/aivan/git/SmartGremlin/background/model"

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.92)
    )

    background = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

    #
    # network
    #
    with tf.Graph().as_default():

        with tf.name_scope('input'):
            ncol = tf.placeholder(dtype=tf.int32, shape=())
            is_train = tf.placeholder(tf.bool, name='is_train')

        # 2D network starts
        layers2d = [tf.random.normal([5,ncol,ncol,n2d_filters])]
        layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
        layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1]))

        # dilated resnet
        dilation = 1
        for _ in range(n2d_layers):
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

        # loss on theta
        logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')
        prob_theta = tf.nn.softmax(logits_theta)

        # loss on phi
        logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')
        prob_phi = tf.nn.softmax(logits_phi)

        # symmetrize
        layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

        # loss on distances
        logits_dist = conv2d(layers2d[-1], 37, 1, padding='SAME')
        prob_dist = tf.nn.softmax(logits_dist)

        # loss on beta-strand pairings
        logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')
        prob_bb = tf.nn.softmax(logits_bb)

        # loss on omega
        logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')
        prob_omega = tf.nn.softmax(logits_omega)

        saver = tf.train.Saver()

        # use ensemble of five networks
        # to do predictions
        with tf.Session(config=config) as sess:
            for ckpt in ['bkgr01', 'bkgr02', 'bkgr03', 'bkgr04', 'bkgr05']:
                saver.restore(sess, BDIR + '/' + ckpt)
                pd, pt, pp, po = sess.run([prob_dist, prob_theta, prob_phi, prob_omega],
                                      feed_dict = {ncol : msa.shape[1], is_train : 0})
                background['dist'].append(pd[0])
                background['theta'].append(pt[0])
                background['omega'].append(po[0])
                background['phi'].append(pp[0])
                print(ckpt, '- done')

    # average predictions over all network
    for key in background.keys():
        background[key] = np.mean(background[key], axis=0)

    return background
