import argparse
import sys

# parser for protein generator
def get_args_hallucinate():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-l", "--len=",  type=int, required=False, dest='LEN', default=100, 
                        help='sequence length')
    parser.add_argument("-s", "--seq=",  type=str, required=False, dest='SEQ', default="",
                        help='starting sequence')

    parser.add_argument('-o', "--ofas=", type=str, required=False, dest='FAS', default='',
                        help='save final sequence to a FASTA files')
    parser.add_argument("--ocsv=", type=str, required=False, dest='CSV', default='',
                        help='save trajectory to a CSV files')

    parser.add_argument("--schedule=",   type=str, required=False, dest='SCHED', default="0.1,40000,2.0,5000",
                        help="simulated annealing schedule: 'T0,n_steps,decrease_factor,decrease_range'")

    parser.add_argument("--trrosetta=",  type=str, required=False, dest='TRDIR', default="../trRosetta/model2019_07",
                        help="path to trRosetta network weights")
    parser.add_argument("--background=", type=str, required=False, dest='BKDIR', default="../background/bkgr2019_05",
                        help="path to background network weights")

    parser.add_argument("--rm_aa=", type=str, required=False, dest='RM_AA', default="",
                        help="disable specific amino acids from being sampled (ex: 'C' or 'W,Y,F')")

    parser.add_argument('--aa_weight=', type=float, required=False, dest='AA_WEIGHT', default=0.0,
                        help='weight for the aa composition biasing loss term')
    
    args = parser.parse_args()

    return args

