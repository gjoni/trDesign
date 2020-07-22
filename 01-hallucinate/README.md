## Usage
```
usage: hallucinate.py [-h] [-l LEN] [-s SEQ] [-o FAS] [--ocsv= CSV]
                      [--schedule= SCHED] [--trrosetta= TRDIR]
                      [--background= BKDIR] [--rm_aa= RM_AA]
                      [--aa_weight= AA_WEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  -l LEN, --len= LEN    sequence length (default: 100)
  -s SEQ, --seq= SEQ    starting sequence (default: )
  -o FAS, --ofas= FAS   save final sequence to a FASTA files (default: )
  --ocsv= CSV           save trajectory to a CSV files (default: )
  --schedule= SCHED     simulated annealing schedule:
                        'T0,n_steps,decrease_factor,decrease_range' (default:
                        0.1,40000,2.0,5000)
  --trrosetta= TRDIR    path to trRosetta network weights (default:
                        ../trRosetta/model2019_07)
  --background= BKDIR   path to background network weights (default:
                        ../background/bkgr2019_05)
  --rm_aa= RM_AA        disable specific amino acids from being sampled (ex:
                        'C' or 'W,Y,F') (default: )
  --aa_weight= AA_WEIGHT
                        weight for the aa composition biasing loss term
                        (default: 0.0)
```

#### Hallucinate a random protein of length 100
```
python ./hallucinate.py -l 100 -o seq.fa
```

#### Hallucinate starting from a given sequence
```
python ./hallucinate.py \
	-s GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE  \  # PDB ID 6MRR
	--schedule=0.05,5000,2,1000 \  # start with T0=0.05, perform 5000 steps, divide T by 2 every 1000 steps
	--rm_aa=C,W \                  # don't sample cysteines and tryptophans
	--aa_weight=1.0 \              # turn on amino acid composition biasing term
	-o seq.fa
```

#### Generate residue-residue distances and orientations using trRosetta
```
python ../trRosetta/network/predict.py -m ../trRosetta/model2019_07 seq.fa seq.npz
```
