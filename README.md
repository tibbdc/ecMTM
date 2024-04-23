# ecMTM
# The process for ecMTM model construction.
## Installation
1.create ECMpy environment using conda:
```
$ conda create -n ECMpy python=3.6.5
```
2.install related packages using pip:
```
$ conda activate ECMpy
$ pip install cobra==0.13.3
$ pip install plotly
$ pip install -U kaleido
$ pip install nbformat
$ pip install requests
$ pip install Bio
$ pip install scipy
$ pip install ipykernel
```
Modeling Process
![Fig1](https://github.com/wangtao-cell/ecMTM/assets/59329042/afaba09a-6b90-4345-b1c8-d55ac2972db6)

kcat from AutoPACMEN:iDL1450_get_data/reaction_kcat_MW.csv
kcat from DLKcat:get_reaction_kcat/kact/DLKcat3.30.cav
kcat from TurNuP:get_reaction_kcat/kact/TurNuP.csv

01_Constuct eciyw1475_AP.ipynb

Construction of eciYW1475_AP.

01_Constuct eciyw1475_DLKcat.ipynb

Construction of eciYW1457_DL.

01_Constuct eciyw1475_TurNup.ipynb

Construction of ecMTM.

02_CDF_kcat_and_mw.ipynb

Cumulative distribution of kcat and molecular weights.
03_FVA.ipynb

Comparative flux variability analysis.

04_PhPP_analysis.ipynb

Phenotype phase plane (PhPP) analysis.

05_trade-off.ipynb

Overflow metabolism simulation.
06_Substrte cascade utilization.ipynb

Substrte cascade utilization prediction.
07_metabolic_engineering_targets.ipynb

Metabolic engineering targets prediction.
