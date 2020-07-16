import tensorflow as tf
from utils import *

def get_background(len,DIR):

    # network params
    n2d_layers   = 36
    n2d_filters  = 64
    window2d     = 3

    activation = tf.nn.elu
    conv1d = tf.layers.conv1d
    conv2d = tf.layers.conv2d

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True)
    )

    background = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

    #
    # network
    #
    with tf.Graph().as_default():

        with tf.name_scope('input'):
            ncol = tf.placeholder(dtype=tf.int32, shape=())

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
            layers2d.append(tf.keras.layers.Dropout(rate=0.15)(layers2d[-1], training=False))
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
            for filename in os.listdir(DIR):
                if not filename.endswith(".index"):
                    continue
                mname = DIR+"/"+os.path.splitext(filename)[0]
                print('reading weights from:', mname)
                saver.restore(sess, mname)
                pd, pt, pp, po = sess.run([prob_dist, prob_theta, prob_phi, prob_omega],
                                      feed_dict = { ncol : len })
                background['dist'].append(np.mean(pd,axis=0))
                background['theta'].append(np.mean(pt,axis=0))
                background['omega'].append(np.mean(po,axis=0))
                background['phi'].append(np.mean(pp,axis=0))
                #print(ckpt, '- done')

    # average predictions over all networks
    for key in background.keys():
        background[key] = np.mean(background[key], axis=0)

    return background
