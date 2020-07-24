##########################
## SUPRESS ALL WARNINGS ##
##########################
import warnings, logging, os
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
##########################

import sys, getopt, subprocess
from subprocess import DEVNULL
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

####################
## load libraries ##
####################
from utils import *
from models import *

def main(argv):
  # parse arguments
  ag = parse_args()
  ag.txt("-------------------------------------------------------------------------------------")
  ag.txt("TrRosetta")
  ag.txt("-------------------------------------------------------------------------------------")
  ag.add(["pdb=",    "p:"], None,  str,   ["load pdb for scoring"])
  ag.add(["chain=",  "c:"], None,  str,   ["specify chain to use"])
  ag.add(["seq=",    "s:"], None,  str,   ["single sequence input (in fasta format)"])
  ag.add(["msa=",    "m:"], None,  str,   ["multiple seq. alignment input (in fasta format)"])
  ag.add(["pssm="],         None,  str,   ["pssm input"])
  ag.txt("--------------------------------------------------------------------------------------")
  ag.add(["cce_cutoff="],   None,  float, ["filter cce to CB ≤ x"])
  ag.add(["list"],          False, None,  ["treat input as list of files to run"])
  ag.add(["save_npz"],      False, None,  ["save data for PyRosetta"])
  ag.add(["save_img"],      False, None,  ["save image of contact map"])
  ag.txt("-------------------------------------------------------------------------------------")
  ag.add(["diag="],         0.4,   float)
  ag.add(["serial"],        False, None,  ["enable approx. serial mode (instead of parallel)"])
  ag.add(["n_models="],     5,     int,   ["number of TrRosetta models to load into memory"])
  ag.txt("-------------------------------------------------------------------------------------")
  o = ag.parse(argv)

  if o.pdb is None and o.seq is None and o.msa is None and o.pssm is None:
    ag.usage(f"ERROR: pdb,fasta,msa or pssm must be defined")

  # default params for [s]etup and [d]esign stage
  s_inputs = {"DB_DIR":DB_DIR, "n_models":o.n_models,
              "serial":o.serial, "diag":o.diag}
  if o.msa is not None:
    s_inputs["msa_design"] = True

  # make model
  model = mk_predict_model(**s_inputs)

  def do(x,mode,pre=None,pdb_out=None):
    if mode == "seq":
      seq = x
      inputs = np.eye(21)[AA_to_N(x)][None,None]
    if mode == "msa":
      nam, seqs = parse_fasta(x, a3m=True)
      msa = mk_msa(seqs)
      seq = seqs[0]
      inputs = msa[None]
    if mode == "pssm":
      pssm_mtx = np.loadtxt(x)
      seq = N_to_AA(pssm_mtx.argmax(-1))[0]
      L,A = pssm_mtx.shape
      if A < 21: pssm_mtx = np.pad(pssm_mtx,[[0,0],[0,21-A]])
      inputs = pssm_mtx[None,None]
    if mode == "pdb":
      pdb_out = prep_input(x, chain=o.chain)
      seq = pdb_out["seq"][0]
      inputs = np.eye(21)[AA_to_N(seq)][None,None]

    feat = model.predict(inputs)[0]
    feats = split_feat(feat)

    if pre is None: pre = x

    # extract pdb features
    if pdb_out is not None:
      pdb_feat = pdb_out["feat"]
      pdb_feat_cce = np.copy(pdb_feat)

      if o.cce_cutoff is not None:
        pdb_dist = pdb_out["dist_ref"]
        pdb_feat_cce[pdb_dist > cce_cutoff] = 0

      pdb_cce_mask = pdb_feat_cce.sum(-1)
      cce = -(pdb_feat_cce*np.log(feat+1e-8)).sum()/pdb_cce_mask.sum()
      acc = get_dist_acc(feat, pdb_feat)
      print(pre, pdb_feat.shape[1], cce, acc)

    if o.save_img:
      plt.figure(figsize=(5,5)); plt.imshow(feats["dist"].argmax(-1))
      plt.savefig(f"{pre}.png", bbox_inches='tight')
      plt.close()

    if o.save_npz:
      np.savez_compressed(f"{pre}.npz",**feats)

  def do_seq(x,pdb_out=None):
    for pre,seq in zip(*parse_fasta(x, a3m=True)):
      do(seq,"seq",pre,pdb_out=pdb_out)

  def do_it(x,mode):
    pdb_out=None
    if mode != "pdb" and o.pdb is not None:
      print("Scoring...")
      pdb_out = prep_input(o.pdb, chain=o.chain)
    if o.list:
      for line in open(x,"r"):
        if mode == "seq": do_seq(line.rstrip(),pdb_out=pdb_out)
        else: do(line.rstrip(),mode,pdb_out=pdb_out)
    else:
      if mode == "seq": do_seq(x,pdb_out=pdb_out)
      else: do(x,mode,pdb_out=pdb_out)

  if o.msa is not None: do_it(o.msa,"msa")
  elif o.seq is not None: do_it(o.seq,"seq")
  elif o.pssm is not None: do_it(o.pssm,"pssm")
  else: do_it(o.pdb,"pdb")

if __name__ == "__main__":
   main(sys.argv[1:])
