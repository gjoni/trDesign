# TrR_for_design (via Gradient Descent)
Coming soon... Manuscript describing method/analysis awaiting bioRxiv.org approval.

![Figure showing method](https://github.com/gjoni/trDesign/raw/master/02-GD/g937.png)

Download Models
```
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/models.zip
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/bkgr_models.zip
unzip -qqo models.zip
unzip -qqo bkgr_models.zip
```
IMPORTANT: Modify `DB_DIR` in `utils.py`, set path to models

Run example
```
python design.py -p 1QYS.pdb -o tmp.txt --save_png
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
-------------------------------------------------------------------------------------
Extras
-------------------------------------------------------------------------------------
--aa_weight=      : weight for aa loss
--rm_aa=          : disable specific amino acids from being sampled
                    ex: 'C' or 'W,Y,F'
--save_img        : save image of contact map
--save_npz        : save data for PyRosetta
-------------------------------------------------------------------------------------
Experimental options
-------------------------------------------------------------------------------------
--pssm_design     : design a PSSM instead of a single sequence
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
