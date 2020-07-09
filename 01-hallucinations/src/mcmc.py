import tensorflow as tf
import numpy as np
import sys
import time

from resnet import *
from utils import *

def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out


def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x


def mcmc(seq0,ref,nsteps,nsave,beta):

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.92)
    )

    MDIR="/home/aivan/git/SmartGremlin/for_paper/models"
    MLIST=['xaa','xab','xac','xad','xae']

    # load networks in RAM
    w,b = [],[]
    beta_,gamma_ = [],[]

    for m in MLIST:
        mname = MDIR+'/model.'+m
        print('model:', mname)

        w.append([
            tf.train.load_variable(mname, 'conv2d/kernel')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
            for i in range(128)])

        b.append([
            tf.train.load_variable(mname, 'conv2d/bias')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
            for i in range(128)])

        beta_.append([
            tf.train.load_variable(mname, 'InstanceNorm/beta')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
            for i in range(123)])

        gamma_.append([
            tf.train.load_variable(mname, 'InstanceNorm/gamma')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
            for i in range(123)])


    # decide whether to optimize for a given topology
    # or to generate brand new ones
    use_bkgrd = True
    if len(ref['dist'].shape) == 2:
        use_bkgrd = False

    L = len(seq0)

    traj = []

    #
    # network
    #
    Activation   = tf.nn.elu
    n2d_layers   = 61
    with tf.Graph().as_default():

        # inputs
        with tf.name_scope('input'):

            ncol = tf.placeholder(dtype=tf.int32, shape=())
            msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))

            if use_bkgrd == True:
                bd = tf.placeholder(dtype=tf.float32, shape=(None,None,None))
                bo = tf.placeholder(dtype=tf.float32, shape=(None,None,None))
                bt = tf.placeholder(dtype=tf.float32, shape=(None,None,None))
                bp = tf.placeholder(dtype=tf.float32, shape=(None,None,None))

            else:
                dist = tf.placeholder(dtype=tf.uint8, shape=(None,None))
                omega = tf.placeholder(dtype=tf.uint8, shape=(None,None))
                theta = tf.placeholder(dtype=tf.uint8, shape=(None,None))
                phi = tf.placeholder(dtype=tf.uint8, shape=(None,None))

            is_train = tf.placeholder(tf.bool, name='is_train')

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(msa, 21, dtype=tf.float32)

        if use_bkgrd == False:
            dist1hot = tf.one_hot(dist, 37, dtype=tf.float32)
            omega1hot = tf.one_hot(omega, 25, dtype=tf.float32)
            theta1hot = tf.one_hot(theta, 25, dtype=tf.float32)
            phi1hot = tf.one_hot(phi, 13, dtype=tf.float32)

        # collect features
        weight = reweight(msa1hot, 0.8)
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weight)
        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,ncol,42])
        f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)
        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

        #
        # network
        #
        layers2d = [[] for _ in range(5)]
        preds = [[] for _ in range(4)] # theta,phi,dist,omega

        for i in range(len(MLIST)):

            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(Activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1

            if use_bkgrd == True:

                # probabilities for theta and phi
                preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
                preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

                # symmetrize
                layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

                # probabilities for dist and omega
                preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
                preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])

            else:

                # activations for theta and phi
                preds[0].append(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0]
                preds[1].append(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0]

                # symmetrize
                layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

                # activations for dist and omega
                preds[2].append(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0]
                preds[3].append(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0]

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)


        # losses
        if use_bkgrd == False:

            # optimize for target topology
            loss_theta = -tf.math.reduce_mean(pt*theta1hot)
            loss_phi = -tf.math.reduce_mean(pp*phi1hot)
            loss_dist = -tf.math.reduce_mean(pd*dist1hot)
            loss_omega = -tf.math.reduce_mean(po*omega1hot)

        else:

            # generate random topology
            loss_dist = -tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd/bd),axis=-1))
            loss_omega = -tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po/bo),axis=-1))
            loss_theta = -tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt/bt),axis=-1))
            loss_phi = -tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp/bp),axis=-1))

            #loss_dist += -0.5*tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd),axis=-1))
            #loss_omega += -0.5*tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po),axis=-1))
            #loss_theta += -0.5*tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt),axis=-1))
            #loss_phi += -0.5*tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp),axis=-1))

        # total loss
        loss = loss_dist + loss_omega + loss_theta + loss_phi


        #saver = tf.train.Saver()
        with tf.Session(config=config) as sess:

            #saver.restore(sess, CHK)

            # initialize with initial sequence
            seq = aa2idx(seq0).copy().reshape([1,L])
            E = 999.9

            # mcmc steps
            for i in range(nsteps):

                # random mutation at random position
                idx = np.random.randint(L)
                a = np.random.randint(20)
                seq_curr = np.copy(seq)
                seq_curr[0,idx] = a

                # probe effect of mutation
                if use_bkgrd == True:
                    E_curr = sess.run(loss, feed_dict = {
                        msa : seq_curr, ncol : L,
                        bd : ref['dist'], bo : ref['omega'], bt : ref['theta'], bp : ref['phi'],
                        is_train : 0})

                else:
                    E_curr = sess.run(loss, feed_dict = {
                        msa : seq_curr, ncol : L,
                        dist : ref['dist'], omega : ref['omega'], theta : ref['theta'], phi : ref['phi'],
                        is_train : 0})

                # Metropolis criterion
                if E_curr < E:
                    seq = np.copy(seq_curr)
                    E = E_curr
                else:
                    if np.exp((E-E_curr)*beta) > np.random.uniform():
                        seq = np.copy(seq_curr)
                        E = E_curr

                if i%nsave==0:
                    aa = idx2aa(seq[0])
                    print("%8d %s %.6f %.1f"%(i, aa, E, beta))
                    sys.stdout.flush()
                    traj.append([i,aa,E])

                if i % 1000 == 0 and i != 0:
                    beta = beta * 2.0

    return traj
