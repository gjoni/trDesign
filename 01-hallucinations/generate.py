import sys,os
import numpy as np
import pandas as pd
import tensorflow as tf

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/src/')

from utils import *
from mcmc import *
from bkgrd import *
from args import get_args_generate

def main():

    ########################################################
    # 0. get inputs
    ########################################################
    params = get_args_generate()


    ########################################################
    # 1. generate background distributions
    ########################################################
    L = params.LENGTH
    seq0 = np.random.randint(20,size=(1,L))

    print("generate background distributions...")
    ref = get_background(seq0)

    for key,val in ref.items():
        s = np.sum(val*np.log(val)) / L / L
        print("S(%s)= %.5f"%(key,s))

    ########################################################
    # 2. run MCMC
    ########################################################
    #PREFIX = '/home/aivan/git/SmartGremlin/for_paper/models/model.'
    #MODELS = ['xaa','xab','xac','xad','xae']

    #chk = PREFIX + MODELS[params.MODEL]

    nsteps = params.NSTEPS
    nsave  = params.NSAVE
    beta   = params.BETA

    seq0 = idx2aa(seq0[0])
    print("mcmc...")
    traj = mcmc(seq0,ref,nsteps,nsave,beta)


    ########################################################
    # 3. save results
    ########################################################
    df = pd.DataFrame(traj, columns = ['step', 'sequence', 'score'])
    df.to_csv(params.ALN, index = None)


if __name__ == '__main__':
    main()
