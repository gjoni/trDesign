# TrR_for_design
Download Models
```
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/models.zip
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/bkgr_models.zip
unzip -qqo models.zip
unzip -qqo bkgr_models.zip
```
Run example
```
python design.py -p 1QYS.pdb -o tmp.txt
```
HELP
```
-------------------------------------------------------------------------------------
TrRosetta for Design
-------------------------------------------------------------------------------------
--len=       -l   : set length for unconstrained design
--out=       -o   : filename prefix for output
--num=       -n   : number of designs
-------------------------------------------------------------------------------------
Backbone Design (if PDB provided)
-------------------------------------------------------------------------------------
--pdb=       -p   : PDB for fixed backbone design
--chain=     -c   : specify chain to use
--mask=      -m   : set positions to rebuild ['start-end:min-max,...']
                    NOTE: the positions are zero-indexed, first position=0
                    use 'start-end' to specify region to rebuild
                    use 'min-max'   to specify length of indel
                    ex: '10-15:0-5' replace pos. 10-15 with variable-loop of length 0-5
                    ex: '10-15:5'   replace pos. 10-15 with fixed-loop of length 5
                    ex: '10-15'     remove pdb cst. from positions 10 to 15
                    ex: '10'        remove pdb cst. from position 10
--sam=       -s   : number of samples per design run [if --mask]
-------------------------------------------------------------------------------------
Extras
-------------------------------------------------------------------------------------
--aa_weight=      : weight for aa loss
--rm_aa=          : disable specific amino acids from being sampled
                    ex: 'C' or 'W,Y,F'
--save_img        : save image of contact map
--save_npz        : save data for PyRosetta
--save_pdb        : use magic to quickly generate PDB structure
--scwrl           : use scwrl to add sidechains [if --save_pdb]
-------------------------------------------------------------------------------------
Experimental options
-------------------------------------------------------------------------------------
--pssm_design     : design a PSSM instead of a single sequence
--pssm_mask       : use PSSM in regions constrained by PDB
--msa_design      : design a MSA instead of a single sequence
--msa_num=        : number of sequences in MSA
--feat_drop=      : dropout rate for features
                    for --msa_design, we recommend 0.8
--cce_cutoff=     : filter cce to CB ≤ x
--spike=          : initialize design from PDB seq
-------------------------------------------------------------------------------------
Optimization settings
-------------------------------------------------------------------------------------
--opt_iter=       : number of iterations
--opt_adam        : use ADAM optimizer
--opt_decay       : use GD+Decay optimizer
--opt_sample      : sample from PSSM instead of taking argmax of PSSM
--serial          : enable approx. serial mode
--n_models=       : number of TrRosetta models to load into memory
-------------------------------------------------------------------------------------
```
