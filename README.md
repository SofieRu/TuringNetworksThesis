# Thesis Research Project

Master Thesis on Developing a computational framework for quantifying the Robustness of synthetic Turing Gene Regulatory Networks

All my code and simulations for this research project. 



## Push Changes to GitHub

How to commit changes in VS Code:

```bash
git status

git add .

git commit -m "your message"

git push origin main

```

## If git push origin main is rejected

This usually means GitHub has newer changes that are not synced yet.

```bash
git status

git pull --rebase origin main

git push origin main
```


## Connect to Imperial VPN

How to connect to the VPN in VS Code

Command + P -> Git: clone 

Command + P -> Remote-SSH:Connect to Host

Enter Imperial password

Go to VS Code Terminal:

```bash
cd ~/TuringNetworksThesis

module load matplotlib/3.9.2-gfbf-2024a

module load SciPy-bundle/2024.05-gfbf-2024a

source ~/venvs/thesis/bin/activate
```

How to run and submit jobs on HPC:

```bash
python TopologyRanking/3954-lhs-3node.py # just for testing

time python TopologyRanking/3954-lhs-3node.py # test before to check how long it will run (set correct waiting time in run.pbs)

qsub run.pbs # submit job to HPC

qstat -u $USER # check status of job
```
