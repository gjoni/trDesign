# ***trRosetta*** for protein design

The package summarizes developments on the use of [***trRosetta***](https://github.com/gjoni/trRosetta) 
structure prediction network for various protein design applications. We provide core codes for the 
following papers:

[01.](https://github.com/gjoni/trDesign/tree/master/01-hallucinate)
I Anishchenko, TM Chidyausiku, S Ovchinnikov, SJ Pellock, D Baker. 
De novo protein design by deep network hallucination. (2020) bioRxiv, doi:10.1101/2020.07.22.211482.
[PDF](https://www.biorxiv.org/content/10.1101/2020.07.22.211482v1.full.pdf)

[02.](https://github.com/gjoni/trDesign/tree/master/02-GD)
C Norn, B Wicky, D Juergens, S Liu, D Kim, B Koepnick, I Anishchenko, Foldit Players, D Baker, S Ovchinnikov.
Protein sequence design by explicit energy landscape optimization. (2020) bioRxiv, doi:10.1101/2020.07.23.218917.
[LINK](https://www.biorxiv.org/content/10.1101/2020.07.23.218917v1)


## Requirements
```tensorflow``` (tested on versions ```1.13``` and ```1.14```)

## Download and installation

```
# download package
git clone --recursive https://github.com/gjoni/trDesign
cd trDesign

# download trRosetta network weights
wget https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2
tar xf model2019_07.tar.bz2 -C trRosetta/

# download background network weights
wget https://files.ipd.uw.edu/pub/trRosetta/bkgr2019_05.tar.bz2
mkdir -p background && tar xf bkgr2019_05.tar.bz2 -C background/
```


## Links

* [***trRosetta*** structure prediction network](https://github.com/gjoni/trRosetta)

* [***trRosetta*** structure prediction server](http://yanglab.nankai.edu.cn/trRosetta/)

* [structure modeling scripts](http://yanglab.nankai.edu.cn/trRosetta/download/) (require [PyRosetta](http://www.pyrosetta.org/))
