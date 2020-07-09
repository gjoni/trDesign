from pyrosetta import *
init("-mute all")

import sys,os
import numpy as np
import pandas as pd
import tensorflow as tf

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/src/')

from utils import *
from mcmc import *
from args import get_args_redesign

def main():

    ########################################################
    # 0. get inputs
    ########################################################
    params = get_args_redesign()


    ########################################################
    # 1. process PDB
    ########################################################
    pose = pose_from_file(params.PDB)
    seq0 = pose.sequence()
    L = len(seq0)

    dist6d,omega6d,theta6d,phi6d = get_neighbors(pose, 20.0)

    nbins=36
    bins = np.linspace(2.0, 20.0, 37)
    bins180 = np.linspace(0.0, np.pi, 13)
    bins360 = np.linspace(-np.pi, np.pi, 25)

    dist6d[dist6d<0.001] = 999.9

    # bin distance matrix
    dbin = np.digitize(dist6d, bins).astype(np.uint8)
    dbin[dbin > nbins] = 0

    # bin omega
    obin = np.digitize(omega6d, bins360).astype(np.uint8)
    obin[dbin == 0] = 0

    # bin theta
    tbin = np.digitize(theta6d, bins360).astype(np.uint8)
    tbin[dbin == 0] = 0

    # bin theta
    pbin = np.digitize(phi6d, bins180).astype(np.uint8)
    pbin[dbin == 0] = 0

    ref = {'dist':dbin, 'omega':obin, 'theta':tbin, 'phi':pbin}


    ########################################################
    # 2. run MCMC
    ########################################################
    PREFIX = '/home/aivan/git/SmartGremlin/for_paper/models/model.'
    MODELS = ['xaa','xab','xac','xad','xae']

    chk = PREFIX + MODELS[params.MODEL]

    nsteps = params.NSTEPS
    nsave  = params.NSAVE
    beta   = params.BETA

    traj = mcmc(seq0,ref,nsteps,nsave,beta,chk)


    ########################################################
    # 3. save results
    ########################################################
    df = pd.DataFrame(traj, columns = ['step', 'sequence', 'score'])
    df.to_csv(params.ALN, index = None)


if __name__ == '__main__':
    main()
