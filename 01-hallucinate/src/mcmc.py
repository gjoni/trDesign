import tensorflow as tf
import numpy as np
import sys
import time

from utils import *

def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out


def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x


def mcmc(DIR,seq0,bkg,schedule,aa_valid,aa_weight):

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True)
    )

    T0,N,coef,M = schedule
    beta = 1./T0
    nsave = 100

    # load networks in RAM
    w,b = [],[]
    beta_,gamma_ = [],[]

    for filename in os.listdir(DIR):
        if not filename.endswith(".index"):
            continue
        mname = DIR+"/"+os.path.splitext(filename)[0]
        print('reading weights from:', mname)

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
            #ncol = tf.placeholder(dtype=tf.int32, shape=())
            msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))

        ncol = tf.shape(msa)[1]
            
        # background distributions
        bd = tf.constant(bkg['dist'], dtype=tf.float32)
        bo = tf.constant(bkg['omega'], dtype=tf.float32)
        bt = tf.constant(bkg['theta'], dtype=tf.float32)
        bp = tf.constant(bkg['phi'], dtype=tf.float32)
        
        # aa bkgr composition in natives
        aa_bkgr = tf.constant(np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                                        0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                                        0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                                        0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271 ]),dtype=tf.float32)

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(msa, 21, dtype=tf.float32)

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

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(4)]

        for i in range(len(w)):

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


            # probabilities for theta and phi
            preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)
        #pb = tf.reduce_mean(tf.stack(preds[4]),axis=0)

        loss_dist = -tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd/bd),axis=-1))
        loss_omega = -tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po/bo),axis=-1))
        loss_theta = -tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt/bt),axis=-1))
        loss_phi = -tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp/bp),axis=-1))

        # aa composition loss
        aa_samp = tf.reduce_sum(msa1hot[0,:,:20], axis=0)/tf.cast(ncol,dtype=tf.float32)+1e-7
        aa_samp = aa_samp/tf.reduce_sum(aa_samp)
        loss_aa = tf.reduce_sum(aa_samp*tf.log(aa_samp/aa_bkgr))

        # total loss
        loss = loss_dist + loss_omega + loss_theta + loss_phi + aa_weight*loss_aa


        #saver = tf.train.Saver()
        with tf.Session(config=config) as sess:

            # initialize with initial sequence
            seq = aa2idx(seq0).copy().reshape([1,L])
            E = 999.9

            # mcmc steps
            for i in range(N+1):

                # random mutation at random position
                idx = np.random.randint(L)
                seq_curr = np.copy(seq)
                seq_curr[0,idx] = np.random.choice(aa_valid)

                # probe effect of mutation
                E_curr = sess.run(loss, feed_dict = { msa : seq_curr })

                # Metropolis criterion
                if E_curr < E:
                    seq = np.copy(seq_curr)
                    E = E_curr
                else:
                    if np.exp((E-E_curr)*beta) > np.random.uniform():
                        seq = np.copy(seq_curr)
                        E = E_curr

                if i%nsave==0 or i==0:
                    aa = idx2aa(seq[0])
                    print("%8d %s %.6f %.1f"%(i, aa, E, beta))
                    sys.stdout.flush()
                    traj.append([i,aa,E])

                if i % M == 0 and i != 0:
                    beta = beta * coef

    return traj,idx2aa(seq[0])
