import tensorflow as tf
import numpy as np
import string
import time
from utils import *

MDIR="/home/aivan/git/SmartGremlin/for_paper/models"
MLIST=['xaa','xab','xac','xad','xae']

MSA="len100.step1000.fa"

a3m = parse_a3m(MSA)
a3m.shape


w,b = [],[]
beta,gamma = [],[]

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

    beta.append([
        tf.train.load_variable(mname, 'InstanceNorm/beta')
        if i==0 else
        tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
        for i in range(123)])

    gamma.append([
        tf.train.load_variable(mname, 'InstanceNorm/gamma')
        if i==0 else
        tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
        for i in range(123)])


wmin         = 0.8
ns           = 21
n2d_layers   = 61

Activation  = tf.nn.elu


def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out

def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x


with tf.Graph().as_default():

    with tf.name_scope('input'):
        ncol = tf.placeholder(dtype=tf.int32, shape=())
        nrow = tf.placeholder(dtype=tf.int32, shape=())
        msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))

    # collect features
    msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32)
    weight = reweight(msa1hot, wmin)
    f1d_seq = msa1hot[0,:,:20]
    f1d_pssm = msa2pssm(msa1hot, weight)
    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])
    f2d_dca = tf.zeros([nrow,ncol,ncol,442], tf.float32)
    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [nrow,ncol,ncol,442+2*42])


    # network
    layers2d = [[] for _ in range(5)]
    preds = [[] for _ in range(4)] # theta,phi,dist,omega

    for i in range(len(MLIST)):

        layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][0],gamma[i][0]))
        layers2d[i].append(Activation(layers2d[i][-1]))

        # resnet
        idx = 1
        dilation = 1
        for _ in range(n2d_layers):
            layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][idx],gamma[i][idx]))
            layers2d[i].append(Activation(layers2d[i][-1]))
            idx += 1
            layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][idx],gamma[i][idx]))
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

    # average over all branches
    pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
    po = tf.reduce_mean(tf.stack(preds[3]),axis=0)

    with tf.Session() as sess:
        for j in range(100):
            tic = time.time()
            out,_,_,_ = sess.run([pd,pt,pp,po], feed_dict={ncol:a3m.shape[1],nrow:a3m.shape[0],msa:a3m})
            toc = time.time()
            print(j, out.shape, toc-tic)


out.shape
