import argparse
import sys

# parser for sequence redesigner
def get_args_redesign():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', type=str, required=True, dest='PDB', help='input PDB file')
    parser.add_argument('-o', type=str, required=True, dest='ALN', help='output file to store generated sequences')
    parser.add_argument('-b', type=float, required=True, dest='BETA', help='inverse temperature for MCMC')
    parser.add_argument('-m', type=int, required=True, dest='MODEL', help='network to use', choices=range(0,5))
    parser.add_argument('-N', type=int, required=True, dest='NSTEPS', help='total number of MCMC steps')
    parser.add_argument('-M', type=int, required=True, dest='NSAVE', help='saving frequency')

    args = parser.parse_args()

    return args


# parser for sequence generator
def get_args_generate():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: allow for inputting a sequence, not just L
    parser.add_argument('-l', type=int, required=True, dest='LENGTH', help='sequence length')
    #parser.add_argument('-s', type=str, required=False, dest='SEQ', help='starting sequence')
    parser.add_argument('-o', type=str, required=True, dest='ALN', help='output file to store generated sequences')
    parser.add_argument('-b', type=float, required=True, dest='BETA', help='inverse temperature for MCMC')
    #parser.add_argument('-m', type=int, required=True, dest='MODEL', help='network to use', choices=range(0,5))
    parser.add_argument('-N', type=int, required=True, dest='NSTEPS', help='total number of MCMC steps')
    parser.add_argument('-M', type=int, required=True, dest='NSAVE', help='saving frequency')

    args = parser.parse_args()

    return args


# parser for sequence scorer
def get_args_score():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', type=str, required=True, dest='CSV', help='input .csv file with sequences')
    parser.add_argument('-o', type=str, required=False, dest='OUTCSV', default='', help='output .csv file with updated scores')
    parser.add_argument('-r', type=str, required=False, dest='PDBREF', default='', help='reference PDB structure (for visualization)')
    parser.add_argument('-onpz', type=str, required=False, dest='NPZ', default='', help='output folder to store .npz files')
    parser.add_argument('-oviz', type=str, required=False, dest='VIZ', default='', help='output folder to store figures')

    args = parser.parse_args()

    if args.OUTCSV=='' and args.NPZ=='' and args.VIZ=='':
        print('Error: either OUTCSV, or NPZ, or VIZ should be specified')
        sys.exit()

    return args


