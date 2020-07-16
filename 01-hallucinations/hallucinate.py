import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/src/')

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import *
from mcmc import *
from bkgrd import *
from args import get_args_hallucinate

def main():

    ########################################################
    # 0. process inputs
    ########################################################

    args = get_args_hallucinate()

    # any residue types to skip during sampling?
    aa_valid = np.arange(20)
    if args.RM_AA != "":
        aa_skip = aa2idx(args.RM_AA.replace(',',''))
        aa_valid = np.setdiff1d(aa_valid, aa_skip)

    # initialize starting sequence 
    if args.SEQ != "":
        L = len(args.SEQ)
        seq0 = args.SEQ
    else:
        L = args.LEN
        seq0 = idx2aa(np.random.choice(aa_valid, L))

    # simulated annealing schedule
    tmp = args.SCHED.split(',')
    schedule = [float(tmp[0]),
                int(tmp[1]),
                float(tmp[2]),
                int(tmp[3])]


    ########################################################
    # 1. generate background distributions
    ########################################################
    print("generating background distributions...")
    bkg = get_background(L,args.BKDIR)


    ########################################################
    # 2. run MCMC
    ########################################################
    print("mcmc...")
    traj,seq = mcmc(args.TRDIR,seq0,bkg,schedule,aa_valid,args.AA_WEIGHT)

    
    ########################################################
    # 3. save results
    ########################################################
    if args.CSV != "":
        df = pd.DataFrame(traj, columns = ['step', 'sequence', 'score'])
        df.to_csv(args.CSV, index = None)

    if args.FAS != "":
        with open(args.FAS,'w') as f:
            f.write(">seq\n%s\n"%(seq))

if __name__ == '__main__':
    main()
