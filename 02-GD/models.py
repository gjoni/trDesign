# supressing warnings
import warnings, logging, os
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

# Should work in both TF1 and TF2
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v1.keras.backend as K1
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Lambda, Layer, Concatenate, Average
import tensorflow.keras.backend as K
tf1.disable_eager_execution()

# HACK to fix compatibility issues with RTX2080
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True

from utils import *
###################################################
# RESNET
###################################################

# custom layers
class instance_norm(Layer):
  def __init__(self, axes=(1,2),trainable=True):
    super(instance_norm, self).__init__()
    self.axes = axes
    self.trainable = trainable
  def build(self, input_shape):
    self.beta  = self.add_weight(name='beta',shape=(input_shape[-1],),
                                 initializer='zeros',trainable=self.trainable)
    self.gamma = self.add_weight(name='gamma',shape=(input_shape[-1],),
                                 initializer='ones',trainable=self.trainable)
  def call(self, inputs):
    mean, variance = tf.nn.moments(inputs, self.axes, keepdims=True)
    return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, 1e-6)

def RESNET(mode="TrR", blocks=12, weights=None, trainable=False, bkg_sample=1):
  ## INPUT ##
  if mode == "TrR":
    inputs = Input((None,None,526)) # (batch,len,len,feat)
    A = inputs
  if mode == "TrR_BKG":
    inputs = Input(shape=[],dtype=tf.int32)
    A = Lambda(lambda x: tf.random.normal([bkg_sample,x[0],x[0],64]))(inputs)

  ex = {"trainable":trainable}
  A = Dense(64, **ex)(A)
  A = instance_norm(**ex)(A)
  A = Activation("elu")(A)

  ## RESNET ##
  def resnet(X, dilation=1, filters=64, win=3):
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(X)
    Y = instance_norm(**ex)(Y)
    Y = Activation("elu")(Y)
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(Y)
    Y = instance_norm(**ex)(Y)
    return Activation("elu")(X+Y)

  for _ in range(blocks):
    for dilation in [1,2,4,8,16]: A = resnet(A, dilation)
  A = resnet(A, dilation=1)

  ## OUTPUT ##
  A_asym    = A
  p_theta   = Dense(25, activation="softmax", **ex)(A_asym)
  p_phi     = Dense(13, activation="softmax", **ex)(A_asym)
  A_sym     = Lambda(lambda x: (x + tf.transpose(x,[0,2,1,3]))/2)(A)
  p_dist    = Dense(37, activation="softmax", **ex)(A_sym)
  p_omega   = Dense(25, activation="softmax", **ex)(A_sym)
  outs      = Concatenate()([p_theta, p_phi, p_dist, p_omega])

  ## MODEL ##
  model = Model(inputs, outs)
  if weights is not None: model.set_weights(weights)
  return model

def load_weights(filename):
  weights = [np.squeeze(w) for w in np.load(filename, allow_pickle=True)]
  # remove weights for beta-beta pairing
  del weights[-4:-2]
  return weights

################################################################################################
# Ivan's TrRosetta Background model for backbone design
################################################################################################
def get_bkg(L, DB_DIR=".", sample=1):
  # get background feat for [L]ength
  K.clear_session()
  K1.set_session(tf1.Session(config=config))
  bkg = {l:[] for l in L}
  bkg_model = RESNET(mode="TrR_BKG", blocks=7, bkg_sample=sample)
  for w in range(1,5):
    weights = load_weights(f"{DB_DIR}/bkgr_models/bkgr0{w}.npy")
    bkg_model.set_weights(weights)
    for l in L:
      bkg[l].append(bkg_model.predict([l] * sample).mean(0))
  return {l:np.mean(bkg[l],axis=0) for l in L}

##############################################################################
# TrR DESIGN
##############################################################################
class mk_design_model:
  ###############################################################################
  # DO SETUP
  ###############################################################################
  def __init__(self, add_pdb=False, add_bkg=False, add_seq_cst=False,
               add_aa_comp_old=False, add_aa_comp=False, add_aa_ref=False, n_models=5, specific_models=None,
               serial=False, diag=0.4, pssm_design=False, msa_design=False, feat_drop=0, eps=1e-8,
               DB_DIR=".", lid=[0.3,18.0], uid=[1,0]):

    self.serial = serial
    self.feat_drop = feat_drop

    # reset graph
    K.clear_session()
    K1.set_session(tf1.Session(config=config))

    # configure inputs
    self.in_label,inputs = [],[],[]
    def add_input(shape, label, dtype=tf.float32):
      inputs.append(Input(shape, batch_size=1, dtype=dtype))
      self.in_label.append(label)
      return inputs[-1][0] if len(shape) == 0 else inputs[-1]

    I = add_input((None,None,20),"I")
    if add_pdb: pdb = add_input((None,None,100),"pdb")
    if add_bkg: bkg = add_input((None,None,100),"bkg")
    if add_seq_cst: seq_cst = add_input((None,20),"seq_cst")

    loss_weights = add_input((None,),"loss_weights")
    sample = add_input([],"sample",tf.bool)
    hard = add_input([],"hard",tf.bool)
    temp = add_input([],"temp",tf.float32)
    train = add_input([],"train",tf.bool)

    ################################
    # input features
    ################################
    def add_gap(x): return tf.pad(x,[[0,0],[0,0],[0,0],[0,1]])
    I_soft, I_hard = categorical(I, temp=temp, hard=hard, sample=sample)
    # configuring input
    if msa_design:
      print("mode: msa design")
      I_feat = MRF(lid=lid,uid=uid)(add_gap(I_hard))
    elif pssm_design:
        print("mode: pssm design")
        I_feat = PSSM(diag=diag)([I_hard,add_gap(I_soft)])
    else:
      print("mode: single sequence design")
      I_feat = PSSM(diag=diag)([I_hard,add_gap(I_hard)])

    # add dropout to features
    if self.feat_drop > 0:
      e = tf.eye(tf.shape(I_feat)[1])[None,:,:,None]
      I_feat_drop = tf.nn.dropout(I_feat,rate=self.feat_drop)
      # exclude dropout at the diagonal
      I_feat_drop = e*I_feat + (1-e)*I_feat_drop
      I_feat = K.switch(train, I_feat_drop, I_feat)

    ################################
    # output features
    ################################
    self.models = []
    tokens = np.array(["xaa","xab","xac","xad","xae"])
    if specific_models is None:
      specific_models = np.arange(n_models)
      
    for token in tokens[specific_models]:
      # load weights (for serial mode) or all models (for parallel mode)
      print(f"loading model: {token}")
      weights = load_weights(f"{DB_DIR}/models/model_{token}.npy")
      if self.serial: self.models.append(weights)
      else: self.models.append(RESNET(weights=weights, mode="TrR")(I_feat))
        
    if self.serial: O_feat = RESNET(mode="TrR")(I_feat)
    else: O_feat = tf.reduce_mean(self.models,0)

    ################################
    # define loss
    ################################
    self.loss_label,loss = [],[]
    def add_loss(term,label):
      loss.append(term)
      self.loss_label.append(label)

    # cross-entropy loss for fixed backbone design
    if add_pdb:
      pdb_loss = -K.sum(pdb*K.log(O_feat+eps),-1)
      add_loss(K.sum(pdb_loss,[-1,-2])/K.sum(pdb,[-1,-2,-3]),"pdb")

    # kl loss for hallucination
    if add_bkg:
      bkg_loss = -0.25*K.sum(O_feat*K.log(O_feat/(bkg+eps)+eps),-1)
      add_loss(K.mean(bkg_loss,[-1,-2]),"bkg")
      
    # add sequence constraint
    if add_seq_cst:
      seq_cst_loss = -K.sum(seq_cst * K.log(I_soft + eps),-1)
      add_loss(K.mean(seq_cst_loss,[-1,-2]),"seq_cst")

    # amino acid composition loss
    if add_aa_ref:
      # experimental
      aa = tf.constant(AA_REF, dtype=tf.float32)
      I_prob = tf.nn.softmax(I,-1)
      aa_loss = K.sum(K.mean(I_prob*aa,[-2,-3]),-1)
      add_loss(aa_loss,"aa")

    elif add_aa_comp:
      # experimental
      aa = tf.constant(AA_COMP, dtype=tf.float32)
      I_prob = tf.nn.softmax(I,-1)
      aa_loss = K.sum(I_prob*K.log(I_prob/(aa+eps)+eps),-1)
      add_loss(K.mean(aa_loss,[-1,-2]),"aa")
      
    elif add_aa_comp_old:
      # ivan's original AA comp loss (from hallucination paper)
      aa = tf.constant(AA_COMP, dtype=tf.float32)
      I_aa = K.mean(I_hard,-2) # mean over length
      aa_loss = K.sum(I_aa*K.log(I_aa/(aa+eps)+eps),-1)
      add_loss(K.mean(aa_loss,-1),"aa")

    if len(loss) > 0:
      ################################
      # define gradients
      ################################
      print(f"The loss function is composed of the following: {self.loss_label}")
      loss = tf.stack(loss,-1) * loss_weights
      grad = Lambda(lambda x: tf.gradients(x[0],x[1])[0])([loss, I])

      ################################
      # define model
      ################################
      self.out_label = ["grad","loss","feat"]
      outputs = [grad,loss,O_feat]
    else:
      self.out_label = ["feat"]
      outputs = [O_feat]
      
    self.model = Model(inputs, outputs)

  ###############################################################################
  # DO DESIGN
  ###############################################################################
  def design(self, inputs, weights=None, num=1, rm_aa=None,
             opt_method="GD", b1=0.9, b2=0.999, opt_iter=100,
             opt_rate=1.0, opt_decay=2.0, verbose=True,
             temp_ini=1.0, temp_decay=0.0, temp_min=0.5,
             hard=True, hard_switch=None,
             sample=False, sample_switch=None,
             return_traj=False, shuf=True):
    
    if weights is None: weights = {} 
    if hard_switch is None: hard_switch = []
    if sample_switch is None: sample_switch = []

    # define length
    if   "pdb" in inputs: L = inputs["pdb"].shape[-2]
    elif "bkg" in inputs: L = inputs["bkg"].shape[-2]
    elif "I" in inputs: L = inputs["I"].shape[-2]

    # initialize
    if "I" not in inputs or inputs["I"] is None:
      inputs["I"] = np.zeros((1,num,L,20))
    if rm_aa is not None:
      inputs["I"][...,rm_aa] = -1e9

    losses, traj = [],[]
    #best_loss, best_I = np.inf,None
    inputs["I"] += np.random.normal(0,0.01,size=inputs["I"].shape)
    mt,vt = 0,0
    
    # optimize
    for k in range(opt_iter):
      
      # softmax gumbel controls
      if k in hard_switch: hard = (hard == False)
      if k in sample_switch: sample = (sample == False)
      temp = np.maximum(temp_ini*np.exp(-temp_decay*(k+1)),temp_min)
      
      # permute input (for msa_design)
      if shuf and num > 0:
        idx = np.random.permutation(np.arange(inputs["I"].shape[1]))
        inputs["I"] = inputs["I"][:,idx]

      # compute loss/gradient
      p = self.predict(inputs, weights=weights, sample=sample, temp=temp, hard=hard, train=True)
      tot_loss = np.sum(p["loss"])
      losses.append(tot_loss)
      if return_traj: traj.append(p)

      # save best result
      #if tot_loss < best_loss:
      #  best_loss, best_I = tot_loss, np.copy(inputs["I"])

      # GD optimizer
      if opt_method == "GD":
        p["grad"] /= np.sqrt(np.square(p["grad"]).sum((-1,-2),keepdims=True)) + 1e-8
        lr = opt_rate * np.sqrt(L)

      # GD optimizer + decay
      if opt_method == "GD_decay":
        p["grad"] /= np.sqrt(np.square(p["grad"]).sum((-1,-2),keepdims=True)) + 1e-8
        lr = opt_rate * np.power(1 - k/opt_iter, opt_decay)

      # ADAM optimizer
      if opt_method == "ADAM":
        mt = b1*mt + (1-b1)*p["grad"]
        vt = b2*vt + (1-b2)*np.square(p["grad"])
        p["grad"] = mt/(np.sqrt(vt) + 1e-8)
        lr = opt_rate * np.sqrt(1-np.power(b2,k+1))/(1-np.power(b1,k+1))

      # update
      inputs["I"] -= lr * p["grad"]

      # report loss
      if verbose and (k+1) % 10 == 0:
        loss = to_dict(self.loss_label, p["loss"][0])
        print(f"{k+1} loss:"+str(loss).replace(' ','')+f" sample:{sample} hard:{hard} temp:{temp}")

    # recompute output
    # if self.feat_drop == 0 and self.sample == False: inputs["I"] = best_I
    p = self.predict(inputs, weights=weights)
    feat          = p["feat"][0]
    loss          = to_dict(self.loss_label, p["loss"][0])
    print("FINAL loss:"+str(loss).replace(' ',''))
    return {"loss":loss, "feat":feat, "I":best_I[0], "losses":losses, "traj":traj}

  ###############################################################################
  def predict(self, inputs, weights=None, sample=False, hard=True, temp=1.0, train=True):
    if weights is None: weights = {}
    # prep inputs
    weights_list = to_list(self.loss_label, weights, 1)
    inputs["loss_weights"] = np.array([weights_list])
    inputs["sample"] = np.array([sample])
    inputs["hard"] = np.array([hard])
    inputs["temp"] = np.array([temp])
    inputs["train"] = np.array([train])
    inputs_list = to_list(self.in_label, inputs)

    if self.serial:
      preds = [[] for _ in range(len(self.model.outputs))]
      for model_weights in self.models:
        self.model.set_weights(model_weights)
        for n,o in enumerate(self.model.predict(inputs_list)):
          preds[n].append(o)
      outputs =  to_dict(self.out_label, [np.mean(pred,0) for pred in preds])
    else:
      outputs = to_dict(self.out_label, self.model.predict(inputs_list))
      
    outputs["I"] = inputs["I"]
    return outputs
  ###############################################################################

##################################################################################
# process input features
##################################################################################
class MRF(Layer):
  def __init__(self, lam=4.5, lid=[0.3,18.0], uid=[1,0], use_entropy=False):
    super(MRF, self).__init__()
    self.lam = lam
    self.use_entropy = use_entropy
    self.lid, self.lid_scale = lid
    self.uid, self.uid_scale = uid

  def call(self, inputs):
    x = inputs[0]
    N,L,A = [tf.shape(x)[k] for k in range(3)]
    F = L*A

    with tf.name_scope('reweight'):
      if self.lid > 0 or self.uid < 1:
        id_len = tf.cast(L, tf.float32)
        id_mtx = tf.tensordot(x,x,[[1,2],[1,2]]) / id_len
        id_mask = []
        # downweight distant sequences
        if self.lid > 0: id_mask.append(tf.sigmoid((self.lid-id_mtx) * self.lid_scale))
        # downweight close sequences
        if self.uid < 1: id_mask.append(tf.sigmoid((id_mtx-self.uid) * self.uid_scale))
        weights = 1.0/(tf.reduce_sum(sum(id_mask),-1) + (self.uid == 1))
      else:
        # give each sequence equal weight
        weights = tf.ones(N)

    with tf.name_scope('covariance'):
      # compute covariance matrix of the msa
      x_flat = tf.reshape(x, (N,F))
      num_points = tf.reduce_sum(weights)
      one = tf.reduce_sum(tf.square(weights))/num_points
      x_mean = tf.reduce_sum(x_flat * weights[:,None], axis=0, keepdims=True) / num_points
      x_flat = (x_flat - x_mean) * tf.sqrt(weights[:,None])
      cov = tf.matmul(tf.transpose(x_flat),x_flat)/(num_points - one)

    with tf.name_scope('inv_convariance'):
      # compute the inverse of the covariance matrix
      I_F = tf.eye(F)
      rm_diag = 1-tf.eye(L)
      cov_reg = cov + I_F * self.lam/tf.sqrt(tf.reduce_sum(weights))
      inv_cov = tf.linalg.inv(cov_reg + tf.random.uniform(tf.shape(cov_reg)) * 1e-8)

      x1 = tf.reshape(inv_cov,(L,A,L,A))
      x2 = tf.transpose(x1, [0,2,1,3])
      features = tf.reshape(x2, (L,L,A*A))

      # extract contacts
      x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * rm_diag
      x3_ap = tf.reduce_sum(x3,0)
      x4 = (x3_ap[None,:] * x3_ap[:,None]) / tf.reduce_sum(x3_ap)
      contacts = (x3 - x4) * rm_diag

      # combine 2D features
      feat_2D = tf.concat([features, contacts[:,:,None]], axis=-1)

    with tf.name_scope('1d_features'):
      # sequence
      x_i = tf.stop_gradient(x[0,:,:20])
      # pssm
      f_i = tf.reduce_sum(weights[:,None,None] * x, axis=0) / tf.reduce_sum(weights)
      # entropy
      if self.use_entropy:
        h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
      else:
        h_i = tf.zeros((L,1))
      # tile and combine 1D features
      feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
      feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
      feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    # combine 1D and 2D features
    feat = tf.concat([feat_1D_tile_A,feat_1D_tile_B,feat_2D],axis=-1)
    return tf.reshape(feat, [1,L,L,442+2*42])

class PSSM(Layer):
  # modified from MRF to only output tiled 1D features
  def __init__(self, diag=0.4, use_entropy=False):
    super(PSSM, self).__init__()
    self.diag = diag
    self.use_entropy = use_entropy
  def call(self, inputs):
    x,y = inputs
    _,_,L,A = [tf.shape(y)[k] for k in range(4)]
    with tf.name_scope('1d_features'):
      # sequence
      x_i = x[0,0,:,:20]
      # pssm
      f_i = y[0,0]
      # entropy
      if self.use_entropy:
        h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
      else:
        h_i = tf.zeros((L,1))
      # tile and combined 1D features
      feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
      feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
      feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    with tf.name_scope('2d_features'):
      ic = self.diag * tf.eye(L*A)
      ic = tf.reshape(ic,(L,A,L,A))
      ic = tf.transpose(ic,(0,2,1,3))
      ic = tf.reshape(ic,(L,L,A*A))
      i0 = tf.zeros([L,L,1])
      feat_2D = tf.concat([ic,i0], axis=-1)

    feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
    return tf.reshape(feat, [1,L,L,442+2*42])

def categorical(y_logits, temp=1.0, sample=False, train=False, hard=True):
  # ref: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

  def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape,minval=0,maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)
  
  def gumbel_softmax_sample(logits): 
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temp,-1)
  
  def one_hot(x):
    y = tf.one_hot(tf.argmax(x,-1),tf.shape(x)[-1])  # argmax
    return tf.stop_gradient(y-x)+x                   # gradient bypass
  
  y_soft = tf.nn.softmax(y_logits/temp,-1)  
  y_soft = K.switch(sample, gumbel_softmax_sample(y_logits), y_soft)    
  y_hard = K.switch(hard, one_hot(y_soft), y_soft)
  return y_soft, y_hard
